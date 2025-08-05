package ipnet

import (
	"encoding/binary"
	"log"
	"net"
	"syscall"
)

// PacketSender は送信に使うraw socketのファイルディスクリプタを保持
type PacketSender struct {
	fd int
}

// NewPacketSender raw socketを作成し初期化して返す
func NewPacketSender() (*PacketSender, error) {
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_RAW, syscall.IPPROTO_RAW)
	if err != nil {
		return nil, err
	}
	// IPヘッダーを自前で作るため
	if err := syscall.SetsockoptInt(fd, syscall.IPPROTO_IP, syscall.IP_HDRINCL, 1); err != nil {
		syscall.Close(fd)
		return nil, err
	}
	return &PacketSender{fd: fd}, nil
}

// Close はソケットクローズ
func (ps *PacketSender) Close() error {
	return syscall.Close(ps.fd)
}

// SendUDPPacket はIP/UDPパケットを作成して送信する
func (ps *PacketSender) SendUDPPacket(srcIP, dstIP net.IP, srcPort, dstPort int, data []byte) error {
	dstAddr := syscall.SockaddrInet4{}
	copy(dstAddr.Addr[:], dstIP.To4())

	totalLength := 20 + 8 + len(data) // IP(20) + UDP(8) + データ

	buf := make([]byte, totalLength)

	buildIPv4Header(buf[:20], srcIP.To4(), dstIP.To4(), totalLength, syscall.IPPROTO_UDP)
	buildUDPHeader(buf[20:28], srcPort, dstPort, data)
	copy(buf[28:], data)

	return syscall.Sendto(ps.fd, buf, 0, &dstAddr)
}

func buildIPv4Header(buf, srcIP, dstIP []byte, totalLen int, protocol int) {
	buf[0] = 0x45                  // version=4, IHL=5
	buf[1] = 0x00                  // DSCP/ECN
	binary.BigEndian.PutUint16(buf[2:4], uint16(totalLen))
	binary.BigEndian.PutUint16(buf[4:6], 0) // ID
	binary.BigEndian.PutUint16(buf[6:8], 0) // Flags + Fragment Offset
	buf[8] = 64                         // TTL
	buf[9] = byte(protocol)              // Protocol (UDP=17)
	// チェックサム計算
	copy(buf[12:16], srcIP)
	copy(buf[16:20], dstIP)
	cs := checksum(buf)
	binary.BigEndian.PutUint16(buf[10:12], cs)
}

func buildUDPHeader(buf []byte, srcPort, dstPort int, data []byte) {
	binary.BigEndian.PutUint16(buf[0:2], uint16(srcPort))
	binary.BigEndian.PutUint16(buf[2:4], uint16(dstPort))
	length := 8 + len(data)
	binary.BigEndian.PutUint16(buf[4:6], uint16(length))
	binary.BigEndian.PutUint16(buf[6:8], 0) // チェックサム省略可
}

func checksum(data []byte) uint16 {
	var sum uint32
	for i := 0; i < len(data)-1; i += 2 {
		sum += uint32(binary.BigEndian.Uint16(data[i : i+2]))
	}
	if len(data)%2 == 1 {
		sum += uint32(data[len(data)-1]) << 8
	}
	for (sum >> 16) > 0 {
		sum = (sum & 0xffff) + (sum >> 16)
	}
	return ^uint16(sum)
}
