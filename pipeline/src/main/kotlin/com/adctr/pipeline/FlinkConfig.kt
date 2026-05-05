package com.adctr.pipeline

/**
 * Configuration for the Flink feature processing pipeline.
 */
data class FlinkConfig(
    val kafkaBootstrapServers: String = System.getenv("KAFKA_BOOTSTRAP_SERVERS") ?: "localhost:9092",
    val impressionTopic: String = System.getenv("KAFKA_IMPRESSION_TOPIC") ?: "ad-impressions",
    val clickTopic: String = System.getenv("KAFKA_CLICK_TOPIC") ?: "ad-clicks",
    val consumerGroup: String = System.getenv("KAFKA_CONSUMER_GROUP") ?: "ctr-feature-pipeline",
    val redisHost: String = System.getenv("REDIS_HOST") ?: "localhost",
    val redisPort: Int = (System.getenv("REDIS_PORT") ?: "6379").toInt(),
    val mongoUri: String = System.getenv("MONGO_URI") ?: "mongodb://localhost:27017",
    val mongoDatabase: String = System.getenv("MONGO_DATABASE") ?: "ctr_features",
    val checkpointIntervalMs: Long = 60_000L,
    val parallelism: Int = (System.getenv("FLINK_PARALLELISM") ?: "2").toInt(),
)
