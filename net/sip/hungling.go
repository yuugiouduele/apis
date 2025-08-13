func HandleSIPRequest(req *SIPRequest) {
    switch req.Method {
    case "INVITE":
        handleInvite(req)
    case "ACK":
        handleAck(req)
    case "BYE":
        handleBye(req)
    case "REGISTER":
        handleRegister(req)
    default:
        fmt.Println("Unknown SIP method:", req.Method)
    }
}

func handleInvite(req *SIPRequest) {
    fmt.Println("Handling INVITE for:", req.URI)
    // 呼進行処理を実装
}

func handleAck(req *SIPRequest) {
    fmt.Println("Handling ACK")
    // 呼開始確認
}

func handleBye(req *SIPRequest) {
    fmt.Println("Handling BYE")
    // 通話終了処理
}

func handleRegister(req *SIPRequest) {
    fmt.Println("Handling REGISTER")
    // 登録処理（ユーザーエージェント登録管理）
}
