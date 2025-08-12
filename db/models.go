package db

import "time"

type User struct {
	ID        uint      `gorm:"primaryKey"`
	Name      string    `gorm:"size:100;not null"`
	Email     string    `gorm:"size:100;uniqueIndex;not null"`
	CreatedAt time.Time
	UpdatedAt time.Time
	Posts     []Post `gorm:"foreignKey:UserID"`
}

type Post struct {
	ID        uint      `gorm:"primaryKey"`
	UserID    uint      `gorm:"index"`
	Title     string    `gorm:"size:200;not null"`
	Content   string    `gorm:"type:text"`
	CreatedAt time.Time
	UpdatedAt time.Time
	User      User `gorm:"constraint:OnUpdate:CASCADE,OnDelete:SET NULL;"`
}
