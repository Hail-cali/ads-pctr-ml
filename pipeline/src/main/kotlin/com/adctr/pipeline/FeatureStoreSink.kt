package com.adctr.pipeline

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.mongodb.client.MongoClients
import com.mongodb.client.MongoCollection
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction
import org.bson.Document
import redis.clients.jedis.JedisPool
import redis.clients.jedis.JedisPoolConfig

/**
 * Redis sink: writes aggregated features for online serving (< 5ms lookup).
 */
class RedisFeatureSink(
    private val redisHost: String,
    private val redisPort: Int,
) : RichSinkFunction<AggregatedFeatures>() {

    @Transient
    private lateinit var jedisPool: JedisPool

    override fun open(parameters: Configuration) {
        val poolConfig = JedisPoolConfig().apply {
            maxTotal = 16
            maxIdle = 8
        }
        jedisPool = JedisPool(poolConfig, redisHost, redisPort)
    }

    private fun ttlForWindow(windowTag: String): Long = when (windowTag) {
        "5m" -> 600L
        "1h" -> 7200L
        "24h" -> 172800L
        else -> 7200L
    }

    override fun invoke(value: AggregatedFeatures, context: Context) {
        jedisPool.resource.use { jedis ->
            val redisKey = "${value.keyType}:${value.key}"
            val fieldPrefix = "${value.keyType}_ctr_${value.windowTag}"
            jedis.hset(redisKey, mapOf(
                "${fieldPrefix}_impressions" to value.impressions.toString(),
                "${fieldPrefix}_clicks" to value.clicks.toString(),
                "${fieldPrefix}_ctr" to "%.6f".format(value.ctr),
                "${fieldPrefix}_updated_at" to value.updatedAt.toString(),
            ))
            jedis.expire(redisKey, ttlForWindow(value.windowTag))
        }
    }

    override fun close() {
        if (::jedisPool.isInitialized) jedisPool.close()
    }
}

/**
 * MongoDB sink: persists raw events for offline model training.
 */
class MongoEventSink(
    private val mongoUri: String,
    private val database: String,
    private val collection: String = "ad_events",
) : RichSinkFunction<AdEvent>() {

    @Transient
    private lateinit var mongoClient: com.mongodb.client.MongoClient
    @Transient
    private lateinit var mongoCollection: MongoCollection<Document>
    @Transient
    private lateinit var mapper: ObjectMapper

    override fun open(parameters: Configuration) {
        mapper = ObjectMapper().registerKotlinModule()
        mongoClient = MongoClients.create(mongoUri)
        mongoCollection = mongoClient.getDatabase(database).getCollection(collection)
    }

    override fun invoke(value: AdEvent, context: Context) {
        val json = mapper.writeValueAsString(value)
        mongoCollection.insertOne(Document.parse(json))
    }

    override fun close() {
        if (::mongoClient.isInitialized) mongoClient.close()
    }
}
