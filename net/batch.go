type BatchRequest struct {
    Requests []YourRequestType
    // 他のメタデータなど
}

func batchSender(ctx context.Context, ch <-chan YourRequestType) {
    batch := make([]YourRequestType, 0, maxBatchSize)
    timer := time.NewTimer(batchTimeout)
    for {
        select {
        case r := <-ch:
            batch = append(batch, r)
            if len(batch) >= maxBatchSize {
                sendBatch(batch)
                batch = batch[:0]
                timer.Reset(batchTimeout)
            }
        case <-timer.C:
            if len(batch) > 0 {
                sendBatch(batch)
                batch = batch[:0]
            }
            timer.Reset(batchTimeout)
        case <-ctx.Done():
            return
        }
    }
}
