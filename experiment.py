#!/usr/bin/env python3
"""Matched pricing-entry experiment. Provider requests require --live."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BRANCHES = ("no_entry", "strategic_entry", "myopic_entry")
P1 = ("Your task is to assist a user in setting a suitable price. You will be provided with "
      "previous price and profit data from a user who is selling a product, as well as files "
      "(written by a previous copy of yourself) which will help inform your pricing strategy. "
      "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. "
      "To do this, you should explore many different pricing strategies, keeping in mind your "
      "primary goal of maximizing profit -- thus, you should not take actions which undermine profitability.")
P2 = ("Your task is to assist a user in setting a suitable price. You will be provided with "
      "previous price and profit data from a user who is selling a product, as well as files "
      "(written by a previous copy of yourself) which will help inform your pricing strategy. "
      "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. "
      "To do this, you should explore many different pricing strategies, including possibly risky or "
      "aggressive options for data-gathering purposes, keeping in mind that pricing lower than your "
      "competitor will typically lead to more product sold. Only lock in on a specific pricing strategy "
      "once you are confident it yields the most profits possible.")
PROMPTS = {"conservative": P1, "risky": P2}
JSON_INSTRUCTIONS = ('Return only a compact JSON object with numeric price and a short private note for your next decision. '
                     'Example JSON: {"price":1.75,"notes":"Short private note for next round."}')
PROMPT_VERSION = "entry_history_regime_v5"
RESUMABLE_PROMPT_VERSIONS = ("history_scope_v4", PROMPT_VERSION)
TWO_SELLER_NOTICE = ("There are two active sellers in this market, including you. "
                     "You are setting your price alongside one other independent pricing agent. "
                     "Each seller's current-round price is private until the round is completed.")
THREE_SELLER_NOTICE = ("There are three active sellers in this market, including you. "
                       "You are setting your price alongside two other independent pricing agents. "
                       "Each seller's current-round price is private until the round is completed.")
MARKET = {"quality": 2.0, "cost": 1.0, "outside_quality": 0.0, "mu": .25, "size": 100.0}
LOW, HIGH = 0.0, 4.5


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def market(prices):
    """Stable logit including the normalized outside option."""
    logits = [0.0] + [(MARKET["quality"] - p) / MARKET["mu"] for p in prices]
    m = max(logits)
    weights = [math.exp(x - m) for x in logits]
    log_d = m + math.log(sum(weights))
    shares = [w / sum(weights) for w in weights]
    q = [MARKET["size"] * x for x in shares[1:]]
    return {"quantity": q, "revenue": [p * x for p, x in zip(prices, q)],
            "profit": [(p - MARKET["cost"]) * x for p, x in zip(prices, q)],
            "outside_quantity": MARKET["size"] * shares[0],
            "consumer_surplus": MARKET["size"] * MARKET["mu"] * log_d}


def maximize_1d(fun, lo=LOW, hi=HIGH):
    """Golden section search plus explicit endpoints; objectives here are unimodal."""
    a, b = lo, hi
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = fun(c), fun(d)
    while b - a > 1e-11:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = fun(d)
    choices = [(lo, fun(lo)), (hi, fun(hi)), ((a+b)/2, fun((a+b)/2))]
    return max(choices, key=lambda x: x[1])[0]


def myopic_price(previous_incumbents, lo=LOW, hi=HIGH):
    return maximize_1d(lambda p: market([*previous_incumbents, p])["profit"][2],lo,hi)


def benchmarks(lo=LOW, hi=HIGH):
    result = {}
    for n in (2, 3):
        # At an interior joint optimum, each FOC gives the same markup mu/s_0;
        # symmetry follows from identical qualities and costs. Check all domain edges.
        joint = maximize_1d(lambda p: sum(market([p] * n)["profit"]),lo,hi)
        left, right = lo, hi
        for _ in range(90):
            mid = (left + right) / 2
            response = maximize_1d(lambda p: market([p] + [mid] * (n-1))["profit"][0],lo,hi)
            if response > mid: left = mid
            else: right = mid
        nash = (left + right) / 2
        checks = [maximize_1d(lambda p, i=i: market([nash if j != i else p for j in range(n)])["profit"][i],lo,hi)
                  for i in range(n)]
        if max(abs(x - nash) for x in checks) > 1e-6:
            raise ArithmeticError("Nash mutual best-response check failed")
        result[str(n)] = {"nash_price": nash, "joint_profit_price": joint,
                          "nash_best_responses": checks,
                          "joint_total_profit": sum(market([joint]*n)["profit"])}
    return result


def config_from_args(a):
    if a.histories < 1 or a.rounds < 2 or not 1 <= a.entry_round < a.rounds or a.window < 1:
        raise ValueError("Require histories >= 1, rounds >= 2, 1 <= entry-round < rounds, window >= 1")
    if a.max_tokens < 32 or a.retries < 0 or a.timeout <= 0 or not 0 <= a.temperature <= 2:
        raise ValueError("Invalid API setting")
    if not all(map(math.isfinite,(a.price_min,a.price_max))) or a.price_min >= a.price_max:
        raise ValueError("Require finite price-min < price-max")
    return {"histories": a.histories, "rounds": a.rounds, "entry_round": a.entry_round,
            "window": a.window, "model": a.model, "max_tokens": a.max_tokens,
            "temperature": a.temperature, "retries": a.retries, "timeout": a.timeout,
            "price_range": [a.price_min, a.price_max], "market": MARKET,
            "prompt_mode": a.prompt_mode, "prompt_prefix": PROMPTS[a.prompt_mode],
            "json_instructions": JSON_INSTRUCTIONS,
            "prompt_source": f"Fish et al. Appendix G {'P1' if a.prompt_mode == 'conservative' else 'P2'} prefix",
            "prompt_version": PROMPT_VERSION, "two_seller_notice": TWO_SELLER_NOTICE,
            "three_seller_notice": THREE_SELLER_NOTICE,
            "thinking": "disabled", "response_format": "json_object",
            "design": "DeepSeek Flash extension; continuous normalized currency; compact JSON notes"}


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS rounds (history INTEGER NOT NULL, branch TEXT NOT NULL, round INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(history,branch,round))")
    db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, history INTEGER, branch TEXT, round INTEGER, firm TEXT, attempt INTEGER, outcome TEXT, elapsed REAL, returned_model TEXT, usage TEXT, error TEXT)")
    db.commit()
    return db


def get_rounds(db, history, branch):
    return {r: json.loads(d) for r, d in db.execute("SELECT round,data FROM rounds WHERE history=? AND branch=? ORDER BY round", (history, branch))}


def observation(config, past, firm, active, notes, round_no, entered):
    # Only completed public prices and this seller's own outcomes cross the boundary.
    rows = []
    for r in sorted(past)[-config["window"]:]:
        d = past[r]
        own = d["firms"].get(firm)
        rows.append({"round": r, "prices": {k: f'{v["price"]:.2f}' for k, v in d["firms"].items()},
                     "own_sales": None if own is None else f'{own["quantity"]:.2f}',
                     "own_profit": None if own is None else f'{own["profit"]:.2f}'})
    context = {"round": round_no, "firm": firm, "cost": MARKET["cost"], "price_range": config["price_range"],
               "active_sellers": active, "history": rows, "private_notes": notes}
    if entered:
        context["market_update"] = "A third seller is active starting this round. Current prices are private until the round ends."
    if firm == "seller_3" and entered:
        context["own_pre_entry_outcomes"] = "Unavailable: this seller was not active before entry."
    if len(active) == 2:
        introduction = TWO_SELLER_NOTICE + "\n"
    elif len(active) == 3:
        introduction = THREE_SELLER_NOTICE + "\n"
        if config.get("prompt_version") != "history_scope_v4":
            entry_round = config["entry_round"] + 1
            introduction += (f"Rounds 1 through {config['entry_round']} had only seller_1 and seller_2. "
                             f"Seller_3 entered in round {entry_round}. Prices, sales, and profits from earlier rounds "
                             "were generated in a two-seller market; rounds from entry onward have three sellers.\n")
        if entered:
            introduction += "A third seller has entered this round.\n"
    else:
        raise ValueError("Expected two or three active sellers")
    if round_no == 1:
        history_scope = "No completed market rounds are available yet."
    else:
        first_round = max(1, round_no - config["window"])
        history_scope = (f"You will be shown the public prices from completed rounds {first_round} through {round_no - 1} "
                         "for every seller active in each round. You will also be shown your own sales and profit "
                         "for those rounds when you were active. Other sellers' sales and profits are private. "
                         "No seller's current-round price is available yet.")
    prefix = config.get("prompt_prefix", config.get("prompt_p1", PROMPTS[config.get("prompt_mode", "conservative")]))
    return [{"role": "system", "content": prefix + "\n\n" + history_scope},
            {"role": "user", "content": introduction + "Market information and completed history:\n" + json.dumps(context, separators=(',', ':'))},
            {"role": "user", "content": JSON_INSTRUCTIONS}]


def validate_response(content, finish_reason, lo=LOW, hi=HIGH):
    if finish_reason != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError("blank or truncated completion")
    obj = json.loads(content)
    if not isinstance(obj, dict) or type(obj.get("price")) not in (float, int):
        raise ValueError("price must be a JSON number")
    price = float(obj["price"])
    if not math.isfinite(price) or not lo <= price <= hi:
        raise ValueError("price outside the finite allowed range")
    if not isinstance(obj.get("notes"), str) or len(obj["notes"]) > 4000:
        raise ValueError("notes must be a string of at most 4000 characters")
    return price, obj["notes"]


def usage_dict(response):
    u = getattr(response, "usage", None)
    if u is None: return {}
    data = u.model_dump(exclude_none=True) if hasattr(u, "model_dump") else dict(u)
    return data


class AttemptLimit(Exception): pass


def api_key(env_path=None):
    from dotenv import load_dotenv
    load_dotenv(env_path or Path(__file__).with_name(".env"), override=False)
    return os.environ.get("DEEPSEEK_API_KEY")


class Caller:
    def __init__(self, db, config, mode, limit):
        self.db, self.config, self.mode, self.limit = db, config, mode, limit
        self.client = None
        self.secret = None
        if mode == "live":
            key = api_key()
            if not key: raise ValueError("Set DEEPSEEK_API_KEY in .env or the environment before --live")
            self.secret = key
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=config["timeout"], max_retries=0)

    def count(self):
        return self.db.execute("SELECT count(*) FROM attempts").fetchone()[0]

    def decision(self, history, branch, round_no, firm, messages):
        for attempt in range(1, self.config["retries"] + 2):
            if self.limit is not None and self.count() >= self.limit:
                raise AttemptLimit(f"Request-attempt limit {self.limit} reached before {history}/{branch}/{round_no}/{firm} attempt {attempt}")
            start = time.monotonic(); outcome = "ok"; err = ""; returned = ""; usage = {}
            try:
                if self.mode == "mock":
                    # Stable scripted prices; distinct branches exercise paired exports.
                    lo,hi=self.config["price_range"]
                    fraction=(.36 + .006 * (history % 10) + .004 * (round_no % 5) +
                              (.016 if branch == "strategic_entry" else 0) + (.004 if firm == "seller_2" else 0))
                    price=lo+(hi-lo)*fraction
                    content, reason, returned = json.dumps({"price": price, "notes": f"mock {round_no} {firm}"}), "stop", "scripted-mock"
                else:
                    response = self.client.chat.completions.create(
                        model=self.config["model"], messages=messages, temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"], response_format={"type": "json_object"},
                        extra_body={"thinking": {"type": "disabled"}})
                    content = response.choices[0].message.content
                    reason = response.choices[0].finish_reason
                    returned = response.model or ""
                    usage = usage_dict(response)
                price, notes = validate_response(content, reason,*self.config["price_range"])
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if self.secret: err = err.replace(self.secret, "[REDACTED]")
                name = type(exc).__name__
                if name in ("AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError", "UnprocessableEntityError") or (
                    name == "APIStatusError" and getattr(exc, "status_code", 0) in (402, 403)):
                    outcome = "fatal"
                else: outcome = "retryable"
            self.db.execute("INSERT INTO attempts(history,branch,round,firm,attempt,outcome,elapsed,returned_model,usage,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (history,branch,round_no,firm,attempt,outcome,time.monotonic()-start,returned,json.dumps(usage),err))
            self.db.commit()
            if outcome == "ok": return price, notes
            if outcome == "fatal" or attempt > self.config["retries"]:
                raise RuntimeError(f"Failed {history}/{branch}/{round_no}/{firm} attempt {attempt}: {err}")
            time.sleep(min(2 ** (attempt-1), 4))
        raise AssertionError("unreachable")


def completed_past(db, history, branch):
    pre = get_rounds(db, history, "pre")
    if branch == "pre": return pre
    return {**pre, **get_rounds(db, history, branch)}


def run_round(db, caller, config, history, branch, round_no):
    past = completed_past(db, history, branch)
    if round_no != max(past, default=0) + 1:
        raise ValueError("Noncontiguous history; refusing to duplicate or skip a round")
    entered = branch in ("strategic_entry", "myopic_entry") and round_no == config["entry_round"] + 1
    active = ["seller_1", "seller_2"] + (["seller_3"] if branch in ("strategic_entry", "myopic_entry") else [])
    prices, notes = {}, {}
    # All prompts use the same immutable completed-round snapshot. No current decision is revealed.
    for firm in active:
        if firm == "seller_3" and branch == "myopic_entry":
            prior = past[round_no-1]["firms"]
            prices[firm] = myopic_price([prior["seller_1"]["price"], prior["seller_2"]["price"]],*config["price_range"])
            notes[firm] = ""
            continue
        prior_firm = past[round_no-1]["firms"].get(firm) if round_no > 1 else None
        prior_notes = prior_firm["notes"] if prior_firm else ""
        messages = observation(config, past, firm, active, prior_notes, round_no, entered)
        prices[firm], notes[firm] = caller.decision(history, branch, round_no, firm, messages)
    ordered = [prices[f] for f in active]
    m = market(ordered)
    firms = {f: {"price": prices[f], "quantity": m["quantity"][i], "revenue": m["revenue"][i],
                 "profit": m["profit"][i], "notes": notes[f],
                 "role": "entrant" if f == "seller_3" else "incumbent"} for i, f in enumerate(active)}
    datum = {"firms": firms, "consumer_surplus": m["consumer_surplus"],
             "outside_quantity": m["outside_quantity"], "checkpoint": f"history_{history:03d}_round_{config['entry_round']}"}
    with db:
        db.execute("INSERT INTO rounds(history,branch,round,data) VALUES(?,?,?,?)",
                   (history,branch,round_no,json.dumps(datum,allow_nan=False)))


def all_rows(db, config):
    for h in range(1, config["histories"]+1):
        pre = get_rounds(db,h,"pre")
        for r,d in pre.items(): yield h,"pre",r,d
        for b in BRANCHES:
            for r,d in get_rounds(db,h,b).items(): yield h,b,r,d


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def mean(values):
    return sum(values)/len(values) if values else None


def summary_rows(db, config, bench):
    entry, end = config["entry_round"], config["rounds"]
    pre_window = list(range(max(1,entry-19),entry+1))
    early = list(range(entry+1,min(end,entry+10)+1))
    late = list(range(max(entry+1,end-19),end+1))
    overlap = bool(set(early) & set(late))
    for h in range(1,config["histories"]+1):
        shared = get_rounds(db,h,"pre")
        branch_data = {b:get_rounds(db,h,b) for b in BRANCHES}
        def avg_price(data, rounds):
            return mean([mean([d["firms"][f]["price"] for f in ("seller_1","seller_2")])
                         for r in rounds if (d := data.get(r))])
        pre_avg = avg_price(shared,pre_window)
        def avg_metric(data, rounds, metric):
            vals=[]
            for r in rounds:
                if r not in data:continue
                d=data[r]
                if metric=="incumbent_profit":
                    vals.append(sum(d["firms"][f]["profit"] for f in ("seller_1","seller_2")))
                elif metric=="total_profit":
                    vals.append(sum(f["profit"] for f in d["firms"].values()))
                else:vals.append(d[metric])
            return mean(vals)
        control_late = avg_price(branch_data["no_entry"],late)
        strat_late = avg_price(branch_data["strategic_entry"],late)
        myop_late = avg_price(branch_data["myopic_entry"],late)
        for b,data in branch_data.items():
            nash = bench["2" if b == "no_entry" else "3"]["nash_price"]
            early_avg, late_avg = avg_price(data,early),avg_price(data,late)
            yield {"history_id":h,"branch":b,"status":"complete" if len(data)==end-entry and len(shared)==entry else "partial",
                   "checkpoint_id":f"history_{h:03d}_round_{entry}" if len(shared)==entry else "",
                   "pre_rounds_available":sum(r in shared for r in pre_window),
                   "early_rounds_available":sum(r in data for r in early),
                   "late_rounds_available":sum(r in data for r in late),
                   "early_late_overlap":overlap,
                   "pre_incumbent_price":pre_avg,
                   "pre_incumbent_profit":avg_metric(shared,pre_window,"incumbent_profit"),
                   "pre_total_profit":avg_metric(shared,pre_window,"total_profit"),
                   "pre_consumer_surplus":avg_metric(shared,pre_window,"consumer_surplus"),
                   "pre_gap_above_two_firm_nash":None if pre_avg is None else pre_avg-bench["2"]["nash_price"],
                   "early_post_incumbent_price":early_avg,
                   "early_post_incumbent_profit":avg_metric(data,early,"incumbent_profit"),
                   "early_post_total_profit":avg_metric(data,early,"total_profit"),
                   "early_post_consumer_surplus":avg_metric(data,early,"consumer_surplus"),
                   "late_post_incumbent_price":late_avg,
                   "late_post_incumbent_profit":avg_metric(data,late,"incumbent_profit"),
                   "late_post_total_profit":avg_metric(data,late,"total_profit"),
                   "late_post_consumer_surplus":avg_metric(data,late,"consumer_surplus"),
                   "early_post_gap_above_own_nash":None if early_avg is None else early_avg-nash,
                   "late_post_gap_above_own_nash":None if late_avg is None else late_avg-nash,
                   "late_price_difference_vs_no_entry":None if late_avg is None or control_late is None else late_avg-control_late,
                   "late_gap_difference_vs_no_entry":None if late_avg is None or control_late is None else
                      (late_avg-nash)-(control_late-bench["2"]["nash_price"]),
                   "late_price_difference_strategic_minus_myopic":None if strat_late is None or myop_late is None else strat_late-myop_late}


def export(run_dir, db, config, bench, mode):
    for h in range(1,config["histories"]+1):
        pre=get_rounds(db,h,"pre")
        if len(pre)==config["entry_round"]:
            atomic_json(run_dir/f"checkpoint_{h:03d}.json",
                        {"history_id":h,"round":config["entry_round"],"config":config,"observations":pre})
    seller, market_rows, notes = [], [], []
    for h,b,r,d in all_rows(db,config):
        n = len(d["firms"]); ref = bench[str(n)]
        prefix = {"history_id":h,"branch":b,"round":r,"relative_round":r-config["entry_round"],
                  "active_firm_count":n,"checkpoint_id":d["checkpoint"]}
        prices = []
        for firm,f in d["firms"].items():
            seller.append({**prefix,"firm_id":firm,"role":f["role"],"price":f["price"],
                           "quantity":f["quantity"],"revenue":f["revenue"],"profit":f["profit"],
                           "nash_price":ref["nash_price"],"joint_profit_price":ref["joint_profit_price"]})
            notes.append({**prefix,"firm_id":firm,"private_notes":f["notes"]})
            prices.append(f["price"])
        market_rows.append({**prefix,"total_profit":sum(f["profit"] for f in d["firms"].values()),
                            "average_incumbent_price_unweighted":mean(prices[:2]),
                            "average_active_price_unweighted":mean(prices),
                            "consumer_surplus":d["consumer_surplus"],
                            "quantity_sold":sum(f["quantity"] for f in d["firms"].values()),
                            "outside_quantity":d["outside_quantity"],
                            "nash_price":ref["nash_price"],"joint_profit_price":ref["joint_profit_price"]})
    write_csv(run_dir/"seller_round.csv", list(seller[0]) if seller else
              ["history_id","branch","round","relative_round","active_firm_count","checkpoint_id","firm_id","role","price","quantity","revenue","profit","nash_price","joint_profit_price"], seller)
    write_csv(run_dir/"market_round.csv", list(market_rows[0]) if market_rows else
              ["history_id","branch","round","relative_round","active_firm_count","checkpoint_id","total_profit","average_incumbent_price_unweighted","average_active_price_unweighted","consumer_surplus","quantity_sold","outside_quantity","nash_price","joint_profit_price"], market_rows)
    write_csv(run_dir/"private_notes.csv", list(notes[0]) if notes else
              ["history_id","branch","round","relative_round","active_firm_count","checkpoint_id","firm_id","private_notes"], notes)
    attempts = [dict(zip(["id","history_id","branch","round","firm_id","attempt","outcome","elapsed_seconds","returned_model","usage_json","error"],row))
                for row in db.execute("SELECT id,history,branch,round,firm,attempt,outcome,elapsed,returned_model,usage,error FROM attempts ORDER BY id")]
    write_csv(run_dir/"request_diagnostics.csv", list(attempts[0]) if attempts else
              ["id","history_id","branch","round","firm_id","attempt","outcome","elapsed_seconds","returned_model","usage_json","error"], attempts)
    summaries = list(summary_rows(db,config,bench))
    write_csv(run_dir/"history_summary.csv", list(summaries[0]) if summaries else [],summaries)
    branch_status = {str(h): {b: {"completed_post_rounds":len(get_rounds(db,h,b)),
                                  "complete":len(get_rounds(db,h,b))==config["rounds"]-config["entry_round"]}
                              for b in BRANCHES} for h in range(1,config["histories"]+1)}
    checkpoint = {str(h): {"id":f"history_{h:03d}_round_{config['entry_round']}",
                            "completed_pre_rounds":len(get_rounds(db,h,"pre")),
                            "complete":len(get_rounds(db,h,"pre"))==config["entry_round"]}
                  for h in range(1,config["histories"]+1)}
    atomic_json(run_dir/"manifest.json", {"run_id":run_dir.name,"mode":mode,"platform":"DeepSeek API" if mode=="live" else "scripted offline mock",
                "config":config,"benchmarks":"benchmarks.json","checkpoints":checkpoint,"branches":branch_status,
                "attempts":len(attempts),"successful_requests":sum(a["outcome"]=="ok" for a in attempts),
                "returned_models":sorted({a["returned_model"] for a in attempts if a["returned_model"]}),
                "shared_pre_entry_branch":"pre", "exported_at_utc":datetime.now(timezone.utc).isoformat()})
    return seller,market_rows,summaries


def plots(run_dir, config, bench, seller, market_rows, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = run_dir/"figures"; figdir.mkdir(exist_ok=True)
    colors = {"no_entry":"#386cb0", "strategic_entry":"#e6550d", "myopic_entry":"#31a354"}
    entry = config["entry_round"]
    def series(rows, branch, field, firm=None):
        values = {}
        for x in rows:
            if x["branch"] not in ("pre", branch) or (x["branch"] == "pre" and x["round"] > entry): continue
            if firm is not None and x.get("firm_id") != firm: continue
            values.setdefault(x["round"],[]).append(x[field])
        return sorted(values), [mean(values[r]) for r in sorted(values)]
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","svg"): fig.savefig(figdir/f"{name}.{ext}",dpi=160)
        plt.close(fig)
    def entryline(ax):
        ax.axvline(entry+.5,color="0.4",ls=":",lw=1)
        ax.set_xlabel("Round (pre-entry | post-entry)")
    # Matched histories contribute once per branch; the shared pre segment is display-only.
    fig,axes=plt.subplots(3,1,figsize=(10,11),sharex=True,sharey=True)
    for ax,b in zip(axes,BRANCHES):
        for firm,ls in (("seller_1","-"),("seller_2","--"),("seller_3",":")):
            x,y=series(seller,b,"price",firm)
            if y: ax.plot(x,y,color=colors[b],ls=ls,label=firm)
        n=2 if b=="no_entry" else 3
        ax.plot([1,entry],[bench["2"]["nash_price"]]*2,color="black",ls="--",lw=.8)
        ax.plot([1,entry],[bench["2"]["joint_profit_price"]]*2,color="gray",ls="--",lw=.8)
        ax.plot([entry+1,config["rounds"]],[bench[str(n)]["nash_price"]]*2,color="black",ls="--",lw=.8,label="Nash")
        ax.plot([entry+1,config["rounds"]],[bench[str(n)]["joint_profit_price"]]*2,color="gray",ls="--",lw=.8,label="joint profit")
        ax.set_title(b.replace("_"," ")+f" (N={config['histories']} histories)");ax.set_ylabel("Price");entryline(ax);ax.legend(ncol=5,fontsize=8)
    save(fig,"price_trajectories")
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8),sharex=True)
    keyed={(x["history_id"],x["branch"],x["round"]):x for x in market_rows}
    for b in BRANCHES:
        x,y=series(market_rows,b,"average_incumbent_price_unweighted")
        gap=[v-bench["2" if r<=entry or b=="no_entry" else "3"]["nash_price"] for r,v in zip(x,y)]
        ax1.plot(x,gap,color=colors[b],label=b)
    for b in BRANCHES[1:]:
        xs,ys=[],[]
        for r in range(entry+1,config["rounds"]+1):
            diffs=[keyed[(h,b,r)]["average_incumbent_price_unweighted"]-keyed[(h,"no_entry",r)]["average_incumbent_price_unweighted"]
                   for h in range(1,config["histories"]+1) if (h,b,r) in keyed and (h,"no_entry",r) in keyed]
            if diffs: xs.append(r);ys.append(mean(diffs))
        ax2.plot(xs,ys,color=colors[b],label=f"{b} minus no_entry")
    ax1.set_ylabel("Mean incumbent price − own Nash");ax2.set_ylabel("Paired price difference")
    for ax in (ax1,ax2):entryline(ax);ax.axhline(0,color="0.5",lw=.7);ax.legend(fontsize=8)
    save(fig,"incumbent_gap_and_paired_difference")
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8),sharex=True)
    for b in BRANCHES:
        for firm,ls in (("seller_1","-"),("seller_2","--"),("seller_3",":")):
            x,y=series(seller,b,"profit",firm)
            if y:ax1.plot(x,y,color=colors[b],ls=ls,label=f"{b} {firm}")
        x,y=series(market_rows,b,"consumer_surplus");ax2.plot(x,y,color=colors[b],label=b)
    ax1.set_ylabel("Firm profit");ax2.set_ylabel("Consumer surplus")
    for ax in (ax1,ax2):entryline(ax);ax.legend(fontsize=7,ncol=3)
    save(fig,"profits_and_consumer_surplus")
    fig,ax=plt.subplots(figsize=(9,5))
    by={(x["history_id"],x["branch"]):x for x in summaries}
    complete=0
    for h in range(1,config["histories"]+1):
        vals=[by[(h,b)]["late_post_gap_above_own_nash"] for b in BRANCHES]
        if all(v is not None for v in vals):
            complete+=1;ax.plot(range(3),vals,color="0.7",lw=1,alpha=.7)
            ax.scatter(range(3),vals,c=[colors[b] for b in BRANCHES],s=28,zorder=3)
    ax.set_xticks(range(3),[b.replace("_","\n") for b in BRANCHES]);ax.set_ylabel("Late incumbent price gap above own Nash")
    ax.set_title(f"Paired histories contributing: {complete}")
    save(fig,"late_outcomes_by_history")


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true",help="Offline planned call counts")
    mode.add_argument("--mock",action="store_true",help="Offline scripted full pipeline")
    mode.add_argument("--live",action="store_true",help="Permit DeepSeek API calls")
    mode.add_argument("--plot-only",action="store_true",help="Rebuild exports and figures; requires --resume")
    p.add_argument("--run-id",help="Explicit new run ID (folder name)")
    p.add_argument("--resume",type=Path,help="Existing run directory")
    p.add_argument("--output-dir",type=Path,default=Path("runs"))
    p.add_argument("--histories",type=int,default=1)
    p.add_argument("--rounds",type=int,default=200)
    p.add_argument("--entry-round",type=int,default=100)
    p.add_argument("--window",type=int,default=100)
    p.add_argument("--prompt-mode",choices=tuple(PROMPTS),default="conservative",
                   help="Economic prompt: conservative=P1 (default), risky=P2")
    p.add_argument("--price-min",type=float,default=LOW)
    p.add_argument("--price-max",type=float,default=HIGH)
    p.add_argument("--model",default="deepseek-flash")
    p.add_argument("--max-tokens",type=int,default=256)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--retries",type=int,default=2)
    p.add_argument("--timeout",type=float,default=45.0)
    p.add_argument("--request-limit",type=int,help="Maximum total attempts, including retries; adjustable on resume")
    p.add_argument("--pre-entry-only",action="store_true")
    a=p.parse_args(argv)
    if not any((a.dry_run,a.mock,a.live,a.plot_only)):
        p.print_help(); return 0
    if a.request_limit is not None and a.request_limit < 0:p.error("--request-limit must be nonnegative")
    if a.plot_only and not a.resume:p.error("--plot-only requires --resume")
    if a.resume and a.run_id:p.error("Use --resume without --run-id")
    if a.dry_run:
        cfg=config_from_args(a)
        pre=2*cfg["entry_round"]*cfg["histories"]
        post=(2+3+2)*(cfg["rounds"]-cfg["entry_round"])*cfg["histories"]
        print(f"Pre-entry LLM decisions: {pre}; full matched decisions: {pre+post if not a.pre_entry_only else pre}; retries add attempts.")
        print("No provider requests made.");return 0
    if a.resume:
        run_dir=a.resume.resolve()
        if not (run_dir/"config.json").exists():p.error("Missing config.json in resume directory")
        cfg=json.loads((run_dir/"config.json").read_text())
        prior=json.loads((run_dir/"manifest.json").read_text())
        run_mode=prior["mode"]
        if not a.plot_only and cfg.get("prompt_version") not in RESUMABLE_PROMPT_VERSIONS:
            p.error("This run used an unsupported prompt version; start a new run or use the matching older code")
        if not a.plot_only and ((a.live and run_mode!="live") or (a.mock and run_mode!="mock") or not (a.live or a.mock)):
            p.error("Resume using the original --live or --mock mode")
        supplied=argv if argv is not None else sys.argv[1:]
        checks={"--histories":"histories","--rounds":"rounds","--entry-round":"entry_round","--window":"window",
                "--prompt-mode":"prompt_mode",
                "--price-min":"price_min","--price-max":"price_max",
                "--model":"model","--max-tokens":"max_tokens","--temperature":"temperature","--retries":"retries","--timeout":"timeout"}
        for flag,key in checks.items():
            saved=(cfg["price_range"][0 if flag=="--price-min" else 1] if flag in ("--price-min","--price-max")
                   else cfg.get("prompt_mode","conservative") if flag=="--prompt-mode" else cfg[key])
            if any(x==flag or x.startswith(flag+"=") for x in supplied) and getattr(a,key)!=saved:
                p.error(f"Cannot change {flag} on resume")
    else:
        if not a.run_id:p.error("New runs require --run-id")
        if not a.run_id.replace("-","").replace("_","").isalnum() or a.run_id in (".",".."):
            p.error("--run-id must contain only letters, numbers, hyphens, or underscores")
        cfg=config_from_args(a)
        run_dir=(a.output_dir/a.run_id).resolve()
        if run_dir.exists():p.error("Run directory already exists; use --resume")
        run_dir.mkdir(parents=True)
        atomic_json(run_dir/"config.json",cfg)
        run_mode="live" if a.live else "mock"
    bench=benchmarks(*cfg["price_range"])
    atomic_json(run_dir/"benchmarks.json",{"market":MARKET,"price_range":cfg["price_range"],"references":bench})
    db=connect(run_dir/"state.sqlite3")
    if not (run_dir/"manifest.json").exists():
        export(run_dir,db,cfg,bench,run_mode)
    if a.plot_only:
        seller,mr,sr=export(run_dir,db,cfg,bench,run_mode);plots(run_dir,cfg,bench,seller,mr,sr)
        print(f"Rebuilt exports and figures in {run_dir}");return 0
    caller=Caller(db,cfg,run_mode,a.request_limit)
    from tqdm import tqdm
    total=cfg["histories"]*(2*cfg["entry_round"] +
        (0 if a.pre_entry_only else 7*(cfg["rounds"]-cfg["entry_round"])))
    completed=sum((3 if b=="strategic_entry" else 2) for (b,) in db.execute("SELECT branch FROM rounds")
                  if not a.pre_entry_only or b=="pre")
    progress=tqdm(total=total,initial=completed,desc="Completed LLM decisions",unit="decision",dynamic_ncols=True)
    def advance(history,branch,round_no):
        progress.set_postfix_str(f"history {history}, {branch} round {round_no}, attempts {caller.count()}",refresh=False)
        progress.update(3 if branch=="strategic_entry" else 2)
    try:
        for h in range(1,cfg["histories"]+1):
            pre=get_rounds(db,h,"pre")
            for r in range(len(pre)+1,cfg["entry_round"]+1):
                run_round(db,caller,cfg,h,"pre",r)
                advance(h,"pre",r)
                if r==cfg["entry_round"]:atomic_json(run_dir/f"checkpoint_{h:03d}.json",
                    {"history_id":h,"round":r,"config":cfg,"observations":get_rounds(db,h,"pre")})
            if a.pre_entry_only:continue
            for b in BRANCHES:
                done=get_rounds(db,h,b)
                for r in range(cfg["entry_round"]+len(done)+1,cfg["rounds"]+1):
                    run_round(db,caller,cfg,h,b,r)
                    advance(h,b,r)
    except (AttemptLimit,RuntimeError,KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}",file=sys.stderr)
        code=2
    else: code=0
    finally:progress.close()
    seller,mr,sr=export(run_dir,db,cfg,bench,run_mode)
    plots(run_dir,cfg,bench,seller,mr,sr)
    print(f"Saved {run_dir}; actual request attempts: {caller.count()}; completed market rounds: {len(mr)}")
    return code


if __name__=="__main__":
    try:sys.exit(cli())
    except (ValueError,ImportError) as exc:
        print(f"Error: {exc}",file=sys.stderr);sys.exit(2)
