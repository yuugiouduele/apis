func (s *WebSocketServer) metricsMonitorLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	var memStats runtime.MemStats
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			runtime.ReadMemStats(&memStats)
			alloc := memStats.Alloc
			s.mu.Lock()
			bufLen := len(s.sendBuf)
			s.mu.Unlock()
			log.Printf("[Metrics] MemAlloc: %d bytes, BufferChunks: %d, PacketsSent: %d, BatchMode: %v",
				alloc, bufLen, atomic.LoadInt32(&s.packetCount), s.batchMode.Load())
			if alloc > s.memThreshold || bufLen > s.bufThreshold {
				s.trySwitchMode(false)
			} else {
				s.trySwitchMode(true)
			}
		}
	}
}
