package db

import (
	"time"

	"gorm.io/gorm"
)

// 組織
type Organization struct {
	ID        uint           `gorm:"primaryKey"`
	Name      string         `gorm:"size:200;not null"`
	CreatedAt time.Time
	UpdatedAt time.Time

	Users []User `gorm:"foreignKey:OrganizationID"`
}

// ユーザー
type User struct {
	ID             uint           `gorm:"primaryKey"`
	OrganizationID uint           `gorm:"index;not null"`
	Name           string         `gorm:"size:100;not null"`
	Email          string         `gorm:"size:200;uniqueIndex;not null"`
	CreatedAt      time.Time
	UpdatedAt      time.Time
	DeletedAt      gorm.DeletedAt `gorm:"index"` // Soft Delete対応

	Roles          []Role          `gorm:"many2many:user_roles;"`
	AccessHistories []AccessHistory `gorm:"foreignKey:UserID"`

	SelfIntroductions []SelfIntroduction `gorm:"foreignKey:UserID"`
	Achievements      []Achievement      `gorm:"foreignKey:UserID"`
	Requests          []Request          `gorm:"foreignKey:UserID"`
	EncryptionKeys    []EncryptionKey    `gorm:"foreignKey:UserID"`
	Streams           []Stream           `gorm:"foreignKey:UserID"` // 共通ストリームインターフェイス用

	// Videos, Images, AudioもStream型でポリモーフィックにできるが単純化のため個別管理も可
	Videos []Video `gorm:"foreignKey:UserID"`
	Images []Image `gorm:"foreignKey:UserID"`
	Audios []Audio `gorm:"foreignKey:UserID"`
}

// 役割（権限）
type Role struct {
	ID        uint      `gorm:"primaryKey"`
	Name      string    `gorm:"size:100;unique;not null"`
	CreatedAt time.Time
	UpdatedAt time.Time

	Users []User `gorm:"many2many:user_roles;"`
}

// アクセス履歴
type AccessHistory struct {
	ID        uint      `gorm:"primaryKey"`
	UserID    uint      `gorm:"index;not null"`
	AccessedAt time.Time
	IP        string    `gorm:"size:45"`
	CreatedAt time.Time
}

// 自己紹介
type SelfIntroduction struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	Content   string         `gorm:"type:text"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 実績
type Achievement struct {
	ID        uint                `gorm:"primaryKey"`
	UserID    uint                `gorm:"index;not null"`
	Title     string              `gorm:"size:200"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt      `gorm:"index"`

	Details []AchievementDetail  `gorm:"foreignKey:AchievementID"`
}

// 実績詳細
type AchievementDetail struct {
	ID            uint      `gorm:"primaryKey"`
	AchievementID uint      `gorm:"index;not null"`
	Description   string    `gorm:"type:text"`
	CreatedAt     time.Time
	DeletedAt     gorm.DeletedAt `gorm:"index"`
}

// 依頼
type Request struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	Content   string         `gorm:"type:text"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 暗号化キー共通管理
type EncryptionKey struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	TargetType string        `gorm:"size:50;not null"` // Ex: "document", "stream", "video" など対象を記録
	TargetID  uint           `gorm:"not null"`        // 対象のID（ポリモーフィック的に利用）
	Key       string         `gorm:"size:512;not null"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 共通履歴テーブル
type History struct {
	ID         uint           `gorm:"primaryKey"`
	UserID     uint           `gorm:"index;not null"`
	TargetType string         `gorm:"size:50;not null"`  // ex: "document", "stream", "ai_model"
	TargetID   uint           `gorm:"not null"`
	Content    string         `gorm:"type:text"`
	CreatedAt  time.Time
	DeletedAt  gorm.DeletedAt `gorm:"index"`
}

// ストリーム共通構造体（必要に応じ継承やinterface実装を追加）
type Stream struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	Type      string         `gorm:"size:50;not null"` // "video","image","audio"
	URL       string         `gorm:"size:500;not null"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 動画
type Video struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	URL       string         `gorm:"size:500;not null"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 画像
type Image struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	URL       string         `gorm:"size:500;not null"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// 音声
type Audio struct {
	ID        uint           `gorm:"primaryKey"`
	UserID    uint           `gorm:"index;not null"`
	URL       string         `gorm:"size:500;not null"`
	CreatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// AIモデル（音声解析、画像解析などのタイプ区分あり）
type AIModel struct {
	ID        uint           `gorm:"primaryKey"`
	Name      string         `gorm:"size:200;not null"`
	Type      string         `gorm:"size:50;not null"` // audio, image, language, numeric, etc
	Version   string         `gorm:"size:50"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

// --- ユーザー・ロール中間テーブルは自動生成されるが、明示定義も可能 ---
// type UserRole struct {
//	  UserID uint `gorm:"primaryKey"`
//    RoleID uint `gorm:"primaryKey"`
// }
