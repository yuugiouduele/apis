package net

func NewAuthService(secretKey string, tokenDuration time.Duration) *AuthService {
	return &AuthService{
		secretKey:     []byte(secretKey),
		tokenDuration: tokenDuration,
	}
}

func (a *AuthService) GenerateToken(userID string) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	claims := jwt.MapClaims{
		"user_id": userID,
		"exp":     time.Now().Add(a.tokenDuration).Unix(),
		"iat":     time.Now().Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(a.secretKey)
}

func (a *AuthService) ValidateToken(tokenString string) (jwt.MapClaims, error) {
	tok, err := jwt.Parse(tokenString, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("invalid signing method")
		}
		return a.secretKey, nil
	})
	if err != nil {
		return nil, err
	}
	if claims, ok := tok.Claims.(jwt.MapClaims); ok && tok.Valid {
		return claims, nil
	}
	return nil, errors.New("invalid token")
}

// ==================== WebSocket サーバ構造体 ====================



func NewWebSocketServer(auth *AuthService, memThreshold uint64, bufThreshold int, cooldown time.Duration) *WebSocketServer {
	s := &WebSocketServer{
		auth:           auth,
		memThreshold:   memThreshold,
		bufThreshold:   bufThreshold,
		switchCooldown: cooldown,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
	s.batchMode.Store(true)
	s.lastSwitch = time.Now()
	return s
}