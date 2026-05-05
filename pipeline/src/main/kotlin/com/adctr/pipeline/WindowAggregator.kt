package com.adctr.pipeline

import org.apache.flink.api.common.functions.AggregateFunction

/**
 * Accumulator for CTR window aggregation.
 */
data class CtrAccumulator(
    var impressions: Long = 0,
    var clicks: Long = 0,
)

/**
 * Aggregates ad events within a window to compute CTR features.
 */
class CtrAggregateFunction : AggregateFunction<AdEvent, CtrAccumulator, CtrAccumulator> {

    override fun createAccumulator(): CtrAccumulator = CtrAccumulator()

    override fun add(event: AdEvent, acc: CtrAccumulator): CtrAccumulator {
        when (event.eventType) {
            "impression" -> acc.impressions++
            "click" -> acc.clicks++
        }
        return acc
    }

    override fun getResult(acc: CtrAccumulator): CtrAccumulator = acc

    override fun merge(a: CtrAccumulator, b: CtrAccumulator): CtrAccumulator {
        return CtrAccumulator(
            impressions = a.impressions + b.impressions,
            clicks = a.clicks + b.clicks,
        )
    }
}
