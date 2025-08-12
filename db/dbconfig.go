package db

import (
	"fmt"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// Config structでホスト、ポート、ユーザ、DB名など設定可能
type Config struct {
	User     string
	Password string
	Host     string
	Port     int
	DBName   string
}

// InitDB : 指定ConfigでDB接続してAutoMigrate実行
func InitDB(cfg Config) (*gorm.DB, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		cfg.User, cfg.Password, cfg.Host, cfg.Port, cfg.DBName)

	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, err
	}

	// モデルを登録（AutoMigrateでテーブル作成）
	err = db.AutoMigrate(&User{}, &Post{})
	if err != nil {
		return nil, err
	}

	return db, nil
}
