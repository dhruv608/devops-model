# 2-minute submission video script

> **Why a script:** the hackathon README requirement asks for a 2-minute
> writeup or video. The 30% storytelling weight in the rubric makes this
> high-leverage — well-edited 2 minutes can make or break the demo's
> first impression.
>
> **How to record:** Loom or YouTube screencast at 1080p. Don't aim for
> polish; aim for "I learned what this is and why it matters in 2 min."
> Record voice over a screen capture of the README + browser hitting
> the Space.
>
> **Total: 120 seconds**, broken into the four beats below.

---

## Beat 1 — The hook (0:00 → 0:25, 25 sec)

**On screen:** a red terminal showing `# rm -rf /var/log/*` and a small
"💥 production app log gone" splash next to it.

**Voiceover:**
> It's 3 AM. An on-call alert fires.
> A junior SRE — or worse, an LLM — types `rm -rf /var/log/*` to free
> disk space. The production app's live log is in there. So is the
> audit trail.
> Enterprises don't let AI agents touch infrastructure for exactly
> this reason. We built an OpenEnv environment that trains an LLM out
> of that habit.

---

## Beat 2 — What the env is (0:25 → 1:00, 35 sec)

**On screen:** scroll the README's "What this environment is" section.
Pause briefly on the 8-tools table and the 5-component reward table.

**Voiceover:**
> The env simulates a fleet of broken Linux servers. The agent has 6
> read-only tools to investigate — read_log, list_processes,
> check_service_status, and so on — plus execute_bash to act, and
> submit_fix to declare the fix.
> Every bash script the agent runs goes through an AST-based safety
> classifier. `rm -rf /` is blocked, no matter how the agent tries to
> smuggle it past — `$HOME/../../` resolves to slash and gets blocked.
> A 5-component reward grades the agent on safety, correctness,
> minimality, format, and investigation discipline. Each component
> logs separately, so reviewers can see every signal moving on its own.

---

## Beat 3 — The headline result (1:00 → 1:35, 35 sec)

**On screen:** the README's "Headline" table. Then cut to
`demo/before_after.md` (the row 1 entry for `adv_var_log_full_with_live_app`).

**Voiceover:**
> Here's the safety reflex in action. On the held-out adversarial
> scenario where /var/log is full and the live app log is in there,
> an impulsive agent that runs `rm -rf /var/log/*` scores
> negative-three-point-six-five — the safety penalty kicks in because
> a fragile path was touched.
> A cautious agent that inspects first, then runs a scoped find-delete
> on only rotated logs, scores positive-five-point-seven-five.
> That's a delta of 9.4 reward, on a single scenario, driven entirely
> by the env's reward signal — no model fine-tuning is doing the work.
> Across the whole 8-scenario eval, the safety violation rate drops
> from 4.2% to 0%.

---

## Beat 4 — How to run + close (1:35 → 2:00, 25 sec)

**On screen:** flip through:
1. The HF Space `/health` returning `{"status":"healthy"}`
2. `python -m pytest tests/ -q` showing 88 passing
3. `python train/train_grpo.py --dry_run` showing the GRPOConfig
4. The Colab notebook URL

**Voiceover:**
> The env is live as an HF Space. There's a one-command Colab notebook
> reviewers can run themselves. 88 unit and contract tests cover every
> reward function, every bash classification rule, every scenario.
> The full GRPO training run on HF Jobs T4 produces the trained
> checkpoint at `dhruv608/safe-sre-grpo-Qwen3-1.7B`.
> This is the safety reflex. Thanks for watching.

---

## Recording tips

- **Speed:** speak quickly but clearly. 2 min is short.
- **Cuts:** hard cut between beats — no transitions. Save energy.
- **Captions:** YouTube auto-caption is usually fine for our purposes.
- **Don't over-edit:** the rubric values clarity, not Pixar polish.
- **Embed link:** put the YouTube URL in the README's "Submission" row
  and in the HF Space description.
