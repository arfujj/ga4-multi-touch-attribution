# Findings memo: which channels drive purchases?

**To:** Marketing Director · **Re:** Google Merchandise Store, 2020-11-01 to 2021-01-31

**Bottom line:** 5,371 purchases worth $340,582 came from 4,848
customer journeys. 37% of them touched more than one channel. For those
journeys, the credit each channel gets depends on the attribution model, and last-click reporting
undervalues the channels that start journeys. The clearest gains are in tracking and the product page,
not in shifting budget.

### 1. Organic Search is worth more than last-click reporting shows
**The number:** Both data-driven models give Organic Search more revenue credit than last touch: $109,173 → $126,801 (Markov, +16.1%) and $120,953 (Shapley, +10.8%); both 95% intervals exclude zero. It opens 41.6% of converting journeys but closes only 34.3%.  
**Why it matters:** Search often starts the journey and another channel closes it, so last-click reports undervalue SEO and content work.  
**Recommended action:** Judge SEO and content on data-driven or position-based credit, and don't cut organic investment based on last-click numbers.

### 2. Direct gets too much credit, and half of it is really missing tracking
**The number:** Direct loses revenue credit under both data-driven models (-4.7% Markov, -13.6% Shapley; both significant). 50% of Direct's 188,830 sessions carry no source information at all, and under last non-direct click its share of conversions falls from 35.8% to 18.3%.  
**Why it matters:** "Direct" is mostly people returning after another channel found them, or visits whose source was lost. It is an outcome, not a lever you can buy.  
**Recommended action:** Fix tracking first: UTM-tag every email and partner link, configure cross-domain measurement and unwanted-referral exclusions, then re-run this analysis.

### 3. Paid Search assists more sales than it gets credit for, but each visit is worth the least
**The number:** Paid Search is the only channel both data-driven models agree last touch under-credits on conversions: 47 → 62.4 (Markov, +32.9%) and 61.1 (Shapley, +29.9%). Even so, it earns $0.33/session vs $1.23 for Organic Search and $2.55 for Referral (Markov).  
**Why it matters:** Last-click ROAS will understate paid search by about a third, but the correction doesn't make it the most efficient channel.  
**Recommended action:** Report paid search on data-driven conversions, then compare cost per data-driven conversion to the other channels before changing its budget.

### 4. Moving budget toward the "under-credited" channel would lose money here
**The number:** Shifting 10% of Email effort (35 sessions) to Organic Search changes data-driven revenue by -$114 over 92 days (95% CI -$208 to -$45), even with constant returns. Email still earns $4.48/session vs $1.23.  
**Why it matters:** Being under-credited is about how a channel is measured, not how efficient it is. Budget decisions need value per unit of spend, and GA4 has no cost data.  
**Recommended action:** Protect Email. Use the attribution results to re-rank channels, then make reallocation decisions with cost data and a holdout test.

### 5. The biggest leak is the product page, not checkout, and it isn't a mobile problem
**The number:** Only 19.7% of sessions that view a product add it to the cart (61,847 of 77,020 sessions drop). Once checkout starts, 64.7% add payment info and 73.8% of those buy. Desktop (19.6%) and mobile (19.9%) add to cart at the same rate (p = 0.28).  
**Why it matters:** Effort spent on checkout UX or a mobile redesign would target the wrong step.  
**Recommended action:** A/B test the product page: price and shipping clarity, stock and size information, and the prominence of the add-to-cart button.

### Limitations
- **Obfuscated sample data.** 26.3% of sessions have no source and are treated as Direct (GA4's convention), and 8.8% fall in Other/Unknown. Google also masks some values as `<Other>` or `(data deleted)`.
- **Short window.** 92 days (2020-11-01 to 2021-01-31), spanning the holidays. Journeys that started before November are cut off, and users who would have bought after January are counted as non-converting.
- **Attribution is correlational.** Every model, including the data-driven ones, redistributes credit for sales that happened. None measures what would have happened without a channel. Only holdout or geo experiments can show incrementality.
- **No cost or cross-device data.** Effort is proxied by sessions, and journeys are per browser (`user_pseudo_id`), so people who switch devices look like separate users.

_Every number in this memo is generated from the pipeline outputs by `src/mta/writeup.py`._
