plugins {
    kotlin("jvm") version "1.9.22"
    id("com.github.johnrengelman.shadow") version "8.1.1"
    application
}

group = "com.adctr"
version = "1.0.0"

val flinkVersion = "1.18.1"
val kafkaVersion = "3.6.1"
val jedisVersion = "5.1.0"
val mongoVersion = "4.11.1"
val jacksonVersion = "2.16.1"
val slf4jVersion = "2.0.11"

repositories {
    mavenCentral()
}

dependencies {
    // Flink
    compileOnly("org.apache.flink:flink-streaming-java:$flinkVersion")
    compileOnly("org.apache.flink:flink-clients:$flinkVersion")
    implementation("org.apache.flink:flink-connector-kafka:3.1.0-1.18")
    implementation("org.apache.flink:flink-connector-base:$flinkVersion")

    // Serialization
    implementation("com.fasterxml.jackson.core:jackson-databind:$jacksonVersion")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:$jacksonVersion")

    // Redis
    implementation("redis.clients:jedis:$jedisVersion")

    // MongoDB
    implementation("org.mongodb:mongodb-driver-sync:$mongoVersion")

    // Logging
    implementation("org.slf4j:slf4j-api:$slf4jVersion")
    runtimeOnly("org.slf4j:slf4j-simple:$slf4jVersion")

    // Kotlin
    implementation(kotlin("stdlib"))

    // Test
    testImplementation(kotlin("test"))
    testImplementation("org.apache.flink:flink-streaming-java:$flinkVersion")
    testImplementation("org.apache.flink:flink-clients:$flinkVersion")
}

application {
    mainClass.set("com.adctr.pipeline.FeatureProcessorKt")
}

tasks.shadowJar {
    archiveBaseName.set("ctr-feature-pipeline")
    archiveClassifier.set("all")
    mergeServiceFiles()
}

kotlin {
    jvmToolchain(17)
}
