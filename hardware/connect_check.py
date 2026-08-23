"""First contact with IBM Quantum Platform: what does this account actually have?

Run this before anything else. It answers the only question that matters at the
start -- whether the CCQC allocation is attached to your account yet, or whether
you are looking at the free Open-plan public devices -- and it costs no QPU time.

    # one time, after copying the API key from the IBM Quantum Platform dashboard
    python connect_check.py --save-token <44-char-api-key>
    # if the CCQC allocation has its own instance, save that too
    python connect_check.py --save-token <key> --save-instance <CRN>

    # every time after that
    python connect_check.py

The distinction to watch for: an Open-plan account sees the public devices with a
small monthly budget shared by everyone, while a dedicated allocation appears as
its own instance with its own usage counter. Submitting the campaign to the wrong
one either fails or quietly spends someone else's minutes.
"""
from __future__ import annotations

import argparse
import sys


def fmt_seconds(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x > 3600:
        return "%.2f h" % (x / 3600)
    if x > 60:
        return "%.1f min" % (x / 60)
    return "%.0f s" % x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-token", default=None, help="API key from the dashboard (saved to disk once)")
    ap.add_argument("--save-instance", default=None, help="instance CRN to save as the default")
    ap.add_argument("--channel", default=None, help="only if support tells you to override it")
    ap.add_argument("--instance", default=None, help="query a specific instance this run")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    if args.save_token:
        kwargs = {"token": args.save_token, "overwrite": True, "set_as_default": True}
        if args.save_instance:
            kwargs["instance"] = args.save_instance
        if args.channel:
            kwargs["channel"] = args.channel
        QiskitRuntimeService.save_account(**kwargs)
        print("saved account credentials to disk\n")

    try:
        saved = QiskitRuntimeService.saved_accounts()
    except Exception as exc:
        saved = {}
        print("could not read saved accounts: %s" % exc)
    if not saved and not args.save_token:
        print("No saved account found.")
        print("Get your API key from the IBM Quantum Platform dashboard, then run:")
        print("    python connect_check.py --save-token <44-char-api-key>")
        return 1
    print("saved account names: %s" % ", ".join(saved) if saved else "(none)")

    kwargs = {}
    if args.channel:
        kwargs["channel"] = args.channel
    if args.instance:
        kwargs["instance"] = args.instance
    try:
        service = QiskitRuntimeService(**kwargs)
    except Exception as exc:
        print("\nCould not connect: %s" % exc)
        print("If the key is right, the instance/CRN may be wrong -- ask CECHelp for the CCQC one.")
        return 1

    acct = service.active_account() or {}
    print("\n" + "=" * 72)
    print("ACCOUNT")
    print("=" * 72)
    print("  channel : %s" % acct.get("channel", "?"))
    print("  instance: %s" % (acct.get("instance") or "(none pinned)"))

    # ---- instances -----------------------------------------------------
    print("\n" + "=" * 72)
    print("INSTANCES (each has its own allocation)")
    print("=" * 72)
    try:
        instances = service.instances()
    except Exception as exc:
        instances = []
        print("  could not list instances: %s" % exc)
    if not instances:
        print("  none reported -- your account may expose a single implicit instance")
    for inst in instances:
        if isinstance(inst, dict):
            name = inst.get("name") or inst.get("crn") or str(inst)
            plan = inst.get("plan") or inst.get("type") or "?"
            print("  - %-46s plan=%s" % (str(name)[:46], plan))
            crn = inst.get("crn")
            if crn:
                print("      CRN: %s" % crn)
        else:
            print("  - %s" % inst)

    try:
        usage = service.usage()
        print("\n  usage on the active instance:")
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)) and ("second" in k or "time" in k):
                print("    %-28s %s" % (k, fmt_seconds(v)))
            elif not isinstance(v, (dict, list)):
                print("    %-28s %s" % (k, v))
        if not usage:
            print("    (empty)")
    except Exception as exc:
        print("\n  usage unavailable: %s" % exc)

    # ---- backends ------------------------------------------------------
    print("\n" + "=" * 72)
    print("BACKENDS VISIBLE TO THIS ACCOUNT")
    print("=" * 72)
    try:
        backends = service.backends()
    except Exception as exc:
        print("  could not list backends: %s" % exc)
        return 1
    if not backends:
        print("  none -- the allocation is probably not attached yet")
        return 1

    print("  %-22s %7s %10s %9s  %s" % ("name", "qubits", "processor", "queued", "status"))
    ccqc = []
    for b in backends:
        try:
            st = b.status()
            queued, operational = st.pending_jobs, st.operational
        except Exception:
            queued, operational = "?", "?"
        try:
            fam = b.configuration().processor_type.get("family", "?")
        except Exception:
            fam = "?"
        print("  %-22s %7s %10s %9s  %s"
              % (b.name, b.num_qubits, fam, queued,
                 "operational" if operational else "DOWN"))
        if any(t in b.name.lower() for t in ("cleveland", "ccqc", "clinic")):
            ccqc.append(b.name)

    # ---- what to do next ------------------------------------------------
    print("\n" + "=" * 72)
    print("NEXT STEP")
    print("=" * 72)
    if ccqc:
        print("  A Cleveland-looking backend is attached: %s" % ", ".join(ccqc))
        print("  Preflight it (still zero QPU cost):")
        print("      python preflight.py --backend %s" % ccqc[0])
    else:
        print("  No backend with a Cleveland/CCQC-looking name is attached to this account.")
        print("  Either the Miami allocation is not linked yet, or it is exposed under a")
        print("  different name -- ask CECHelp@MiamiOH.edu for the exact backend name and CRN.")
        print("  Meanwhile you can rehearse the whole campaign for free against any device")
        print("  above, or with no credentials at all:")
        print("      python preflight.py --backend %s" % backends[0].name)
        print("      python preflight.py --fake")
    print("\n  Do NOT submit anything until preflight reports that every schedule places")
    print("  its pulse centres exactly on the grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
