package com.adctr.pipeline

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import org.apache.flink.api.common.eventtime.WatermarkStrategy
import org.apache.flink.api.common.serialization.AbstractDeserializationSchema
import org.apache.flink.connector.kafka.source.KafkaSource
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer
import org.apache.kafka.clients.consumer.OffsetResetStrategy
import org.apache.flink.streaming.api.CheckpointingMode
import org.apache.flink.streaming.api.datastream.DataStream
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.slf4j.LoggerFactory
import java.time.Duration

private val logger = LoggerFactory.getLogger("FeatureProcessor")

/**
 * JSON deserializer for AdEvent.
 */
class AdEventDeserializer : AbstractDeserializationSchema<AdEvent>() {
    private val mapper = ObjectMapper().registerKotlinModule()

    override fun deserialize(message: ByteArray): AdEvent {
        return mapper.readValue(message)
    }
}

/**
 * Builds a Kafka source for ad events (impressions + clicks merged).
 */
fun buildKafkaSource(config: FlinkConfig): KafkaSource<AdEvent> {
    return KafkaSource.builder<AdEvent>()
        .setBootstrapServers(config.kafkaBootstrapServers)
        .setTopics(config.impressionTopic, config.clickTopic)
        .setGroupId(config.consumerGroup)
        .setStartingOffsets(OffsetsInitializer.committedOffsets(OffsetResetStrategy.EARLIEST))
        .setValueOnlyDeserializer(AdEventDeserializer())
        .build()
}

/**
 * Applies sliding window aggregation and emits AggregatedFeatures.
 */
fun applyWindowedAggregation(
    stream: DataStream<AdEvent>,
    keySelector: (AdEvent) -> String,
    keyType: String,
    windowSize: Time,
    slideSize: Time,
    windowTag: String,
): DataStream<AggregatedFeatures> {
    return stream
        .keyBy { keySelector(it) }
        .window(SlidingEventTimeWindows.of(windowSize, slideSize))
        .aggregate(
            CtrAggregateFunction(),
            { key: String, window: TimeWindow, results: Iterable<CtrAccumulator>, out: Collector<AggregatedFeatures> ->
                val acc = results.first()
                val ctr = if (acc.impressions > 0) acc.clicks.toDouble() / acc.impressions else 0.0
                out.collect(
                    AggregatedFeatures(
                        key = key,
                        keyType = keyType,
                        windowTag = windowTag,
                        impressions = acc.impressions,
                        clicks = acc.clicks,
                        ctr = ctr,
                        updatedAt = window.end,
                    )
                )
            }
        )
}

/**
 * Main Flink job: consumes ad events from Kafka, computes windowed CTR features,
 * writes to Redis (online) and MongoDB (offline).
 */
fun main() {
    val config = FlinkConfig()
    logger.info("Starting CTR Feature Processor with config: $config")

    val env = StreamExecutionEnvironment.getExecutionEnvironment().apply {
        parallelism = config.parallelism
        enableCheckpointing(config.checkpointIntervalMs, CheckpointingMode.EXACTLY_ONCE)
    }

    // Kafka source
    val kafkaSource = buildKafkaSource(config)
    val eventStream = env.fromSource(
        kafkaSource,
        WatermarkStrategy.forBoundedOutOfOrderness<AdEvent>(Duration.ofSeconds(5))
            .withTimestampAssigner { event, _ -> event.timestamp },
        "kafka-ad-events"
    )

    // MongoDB sink for raw events (offline training data)
    eventStream.addSink(MongoEventSink(config.mongoUri, config.mongoDatabase))
        .name("mongo-event-sink")

    // Redis sink for aggregated features (online serving)
    val redisSink = { RedisFeatureSink(config.redisHost, config.redisPort) }

    // User CTR — 5min window, 1min slide
    applyWindowedAggregation(eventStream, { it.userId }, "user", Time.minutes(5), Time.minutes(1), "5m")
        .addSink(redisSink()).name("redis-user-ctr-5m")

    // User CTR — 1h window, 5min slide
    applyWindowedAggregation(eventStream, { it.userId }, "user", Time.hours(1), Time.minutes(5), "1h")
        .addSink(redisSink()).name("redis-user-ctr-1h")

    // User impressions — 24h window, 30min slide
    applyWindowedAggregation(eventStream, { it.userId }, "user", Time.hours(24), Time.minutes(30), "24h")
        .addSink(redisSink()).name("redis-user-ctr-24h")

    // Ad CTR — 1h window, 5min slide
    applyWindowedAggregation(eventStream, { it.adId }, "ad", Time.hours(1), Time.minutes(5), "1h")
        .addSink(redisSink()).name("redis-ad-ctr-1h")

    // Ad CTR — 24h window, 30min slide
    applyWindowedAggregation(eventStream, { it.adId }, "ad", Time.hours(24), Time.minutes(30), "24h")
        .addSink(redisSink()).name("redis-ad-ctr-24h")

    // Slot CTR — 1h window, 5min slide
    applyWindowedAggregation(eventStream, { it.slot }, "slot", Time.hours(1), Time.minutes(5), "1h")
        .addSink(redisSink()).name("redis-slot-ctr-1h")

    env.execute("CTR Feature Processor")
}
