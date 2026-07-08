# ======== ML vs. LLM in Pipelines ========

# ===== Question 1
"""
The ML classifier produces a binary prediction (good or skip) from numeric features. The LLM
produces natural language text, the written recommendation. ML is built for fast consistent
numeric classification, while the LLM is built to generate human readable language.

If you swapped them, using the LLM for the binary prediction would be slower, cost more, and be
less consistent, and could return something outside the two allowed labels. Using the ML model to
write the recommendation would not work since a classifier only outputs a label, not sentences.
"""

# ===== Question 2
# Converting a date string like "2023-07-04" to day-of-week
"""Code since its date parsing."""
# Classifying a job posting as entry-level, mid-level, or senior from freeform text
"""LLM since it requires reading comprehension which a regex can not do."""
# Predicting customer churn given 15 numeric features and a labeled training dataset
"""Trained ML model since that is what a classifier trained on labeled numeric data is for."""
# Normalizing inconsistent city names to a canonical form
"""LLM since the input is irregular and needs language understanding."""
# Summing a column of revenue figures
"""Code since its calculating."""

# ===== Question 3
"""
Incremental processing means only processing the new or changed records each run instead of
redoing the whole dataset every time. It matters here because the pipeline calls the LLM per
record and those calls cost money and time. You dont want to pay for the same records over and over again if they have not changed.

If the transform reprocessed all 365 records every time, the cost would multiply since you pay for
365 LLM calls each run instead of just the new ones, the runtime would grow, and you risk
overwriting or duplicating results that were already correct.
"""

# ======== Prompt  ========

# ===== Question 1
"""
SYSTEM_PROMPT = (
    "You are giving an outdoor running recommendation based on weather conditions."
    "Reply with exactly two sentences. The first sentence states the prediction (good or skip). "
    "The second sentence explains the reasoning based on the temperature and precipitation."
)

For validation, instead of checking the output is one word in a fixed set, I would check that the
response has two sentences and pull the prediction from the first sentence (check if it contains
"good" or "skip"). The validation becomes about format and sentence structure instead of an exact one word match.
"""

# ===== Question 2
import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

# When you would use this in a production pipeline:
"""
You would use this to handle transitional failures like network,rate limits, or a momentary
API outage, so one temporary error does not crash the whole run. Returning None on final failure
lets the calling code handle the bad record or failure easier instead of stopping.
"""