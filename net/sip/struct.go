type CallSession struct {
    CallID    string
    From      string
    To        string
    State     string // 例: "Calling", "Confirmed", "Terminated"
    RemoteAddr *net.UDPAddr
    // 必要に応じてRTP情報やタイマーも保持
}

var sessionTable = make(map[string]*CallSession)

// セッション追加例
func AddSession(callID string, session *CallSession) {
    sessionTable[callID] = session
}
// セッション検索例
func GetSession(callID string) (*CallSession, bool) {
    s, ok := sessionTable[callID]
    return s, ok
}
// セッション削除例
func RemoveSession(callID string) {
    delete(sessionTable, callID)
}
