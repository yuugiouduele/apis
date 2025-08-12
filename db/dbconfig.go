package db

import (
	"fmt"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// PostgreSQL接続設定
type Config struct {
	User     string
	Password string
	Host     string
	Port     int
	DBName   string
	SSLMode  string // 例: "disable"
	TimeZone string // 例: "Asia/Tokyo"
}

// InitDB は複数モデルを一括でAutoMigrateしてDB接続を返す
func InitDB(cfg Config) (*gorm.DB, error) {
	dsn := fmt.Sprintf(
		"host=%s user=%s password=%s dbname=%s port=%d sslmode=%s TimeZone=%s",
		cfg.Host, cfg.User, cfg.Password, cfg.DBName, cfg.Port, cfg.SSLMode, cfg.TimeZone,
	)
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, err
	}

	// ここにモデルを一括登録（interface{}型のスライス展開で）
	models := []interface{}{
		&Organization{}, &User{}, &Role{}, &AccessHistory{},
		&SelfIntroduction{}, &Achievement{}, &AchievementDetail{}, &Request{}, &EncryptionKey{}, &History{},
		&Stream{}, &Video{}, &Image{}, &Audio{},
		&AIModel{},
	}

	err = db.AutoMigrate(models...)
	if err != nil {
		return nil, err
	}

	return db, nil
}
