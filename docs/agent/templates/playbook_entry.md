# Playbook entry: template

One entry per problem, at most half a page, in your own words and any language.
Below the template is an example from our own data.

---

# [step]: [the symptom in a few words]

**Where it shows:** which page, plot, number or message.

**How to confirm:** what to look at or count to be sure it is this problem.

**Causes, common to rare:**
1.
2.

**What to try, cheap to expensive** — and what you expect to see if it works:
1.
2.

**What not to do:**

**When to stop and ask someone:**

**Source:** the dataset, series, date and person this is based on.

---

# Example — CTF estimation: only the views near zero tilt fit

**Where it shows:** the CTF page of the report. Beyond about ±12° the fitted
resolution sits at the 30 Å end of the fit range. On the views that do fit,
defocus changes steadily with tilt angle.

**How to confirm:** count the views fitted better than 20 Å, and plot defocus
against tilt angle for the fitted views. A steady trend — here 0.025–0.035 µm per
degree — means the defocus at high tilt lies outside the range searched for each
view.

**Causes, common to rare:**
1. The defocus search window for each view (±1000 Å by default) is narrower than
   the change in defocus across the tilt series.
2. Thickness or accumulated dose at high tilt. In that case a wider window does
   not help.

**What to try, cheap to expensive:**
1. Wider windows, ±5000 and ±10000 Å, on one series, as separate variants that do
   not overwrite the main result. If cause 1 is right, more views fit, and their
   defocus continues the same trend.
2. Not tested here: check whether the high-tilt power spectra show Thon rings at
   all, and consider excluding the views that do not.

**What not to do:** rerun the whole lane before the one-series test; count views
that stop at the 30 Å bound as fitted.

**When to stop and ask someone:** if fewer than half the views fit even at
±10000 Å.

**Source:** RyR1 dataset (MDC), series 039, October 2026. At ±1000, ±5000 and
±10000 Å, 9, 22 and 26 of 29 views fitted better than 20 Å.
