type WebSocketServer struct {
	conn           *websocket.Conn
	auth           *AuthService
	sendBuf        [][]byte
	mu             sync.Mutex
	batchMode      atomic.Bool
	memThreshold   uint64
	bufThreshold   int
	packetCount    int32
	lastSwitch     time.Time
	switchCooldown time.Duration
	upgrader       websocket.Upgrader
}

type AuthService struct {
	secretKey     []byte
	tokenDuration time.Duration
	mu            sync.Mutex
}