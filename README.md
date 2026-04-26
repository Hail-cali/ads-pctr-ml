# Ad pCTR Pipeline

`from @Hail`

- CTR(Click-Through Rate) 예측 시스템. 오프라인 학습부터 온라인 서빙
- 실시간 feature engineering까지 포함하는 end-to-end ML 파이프라인

---

## Architecture

```
                          +-----------------+
                          |   Kafka Topics  |
                          | ad-impressions  |
                          | ad-clicks       |
                          +--------+--------+
                                   |
                    +--------------+--------------+
                    |                             |
           +--------v--------+           +--------v--------+
           | Flink Pipeline  |           |  MongoDB Sink   |
           | (Kotlin)        |           |  (raw events)   |
           |                 |           +-----------------+
           | Sliding Windows |
           | 5m / 1h / 24h  |
           +--------+--------+
                    |
           +--------v--------+
           |     Redis       |
           | (online features|
           |  < 5ms lookup)  |
           +--------+--------+
                    |
           +--------v--------+      +-----------------+
           |  FastAPI Server |<-----| ONNX Runtime    |
           |  /predict       |      | (DeepFM model)  |
           |  /predict/v2    |      +-----------------+
           |  /predict/batch |
           +--------+--------+
                    |
           +--------v--------+
           |  Prometheus     |-----> Grafana Dashboard
           |  /metrics       |
           +-----------------+
```

### Offline (학습)

```
Criteo TSV  -->  CriteoPreprocessor  -->  PyTorch 학습  -->  ONNX Export
(1M rows)        (standardize +           (AMP, Early      (sigmoid 포함)
                  field-offset hash)        Stopping)
```

### Online (서빙)

```
Request  -->  Redis Feature Lookup  -->  ONNX Inference  -->  CTR 확률 응답
              (Flink이 집계한                (< 10ms)
               user/ad CTR stats)
```