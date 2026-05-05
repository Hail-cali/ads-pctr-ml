package com.adctr.pipeline

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.annotation.JsonProperty

@JsonIgnoreProperties(ignoreUnknown = true)
data class AdEvent(
    @JsonProperty("event_id") val eventId: String = "",
    @JsonProperty("event_type") val eventType: String = "",
    @JsonProperty("user_id") val userId: String = "",
    @JsonProperty("ad_id") val adId: String = "",
    val slot: String = "",
    val context: EventContext = EventContext(),
    val timestamp: Long = 0L,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class EventContext(
    val device: String = "",
    val os: String = "",
    val geo: String = "",
    val hour: Int = 0,
    @JsonProperty("day_of_week") val dayOfWeek: Int = 0,
)

/**
 * Aggregated features computed by Flink windowed operations.
 */
data class AggregatedFeatures(
    val key: String = "",        // e.g. "user:U00001" or "ad:A0001"
    val keyType: String = "",    // "user", "ad", "slot"
    val windowTag: String = "",  // "5m", "1h", "24h"
    val impressions: Long = 0,
    val clicks: Long = 0,
    val ctr: Double = 0.0,
    val updatedAt: Long = System.currentTimeMillis(),
)
