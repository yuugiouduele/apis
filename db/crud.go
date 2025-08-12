package db

import (
	"gorm.io/gorm"
	"fmt"
)

// UserとRoleの多対多Joinでユーザーの役割一覧を取得する例
func GetUserRoles(db *gorm.DB, userID uint) ([]Role, error) {
	var roles []Role
	err := db.Model(&User{ID: userID}).Association("Roles").Find(&roles)
	return roles, err
}

// UserとAccessHistoryをJoinしてユーザのアクセス記録を取得
func GetUserAccessHistories(db *gorm.DB, userID uint) ([]AccessHistory, error) {
	var histories []AccessHistory
	err := db.Where("user_id = ?", userID).Find(&histories).Error
	return histories, err
}

// AchievementとAchievementDetailを一緒に取得（プリロード）
func GetAchievementWithDetails(db *gorm.DB, achievementID uint) (Achievement, error) {
	var achievement Achievement
	err := db.Preload("Details").First(&achievement, achievementID).Error
	return achievement, err
}

// カスタムJoin例：User、Organization、Roleを結合してユーザーデータ取得
func SelectUserWithOrgAndRoles(db *gorm.DB, userID uint) (User, error) {
	var user User
	// プリロードでRoles, Organizationをまとめて取得（複数リレーションの例）
	err := db.Preload("Roles").Preload("Organization").First(&user, userID).Error
	return user, err
}

// 動画とそれに紐づく暗号化情報をJoinで取得（例）
func GetVideosWithEncryptionKeys(db *gorm.DB, userID uint) ([]struct {
	Video          Video
	EncryptionKey  EncryptionKey
}, error) {
	var results []struct {
		Video         Video
		EncryptionKey EncryptionKey
	}
	err := db.Table("videos").
		Select("videos.*, encryption_keys.key").
		Joins("left join encryption_keys on encryption_keys.target_type = ? AND encryption_keys.target_id = videos.id", "video").
		Where("videos.user_id = ?", userID).
		Scan(&results).Error
	return results, err
}

// 複合的なJoinやフィルター条件もgormで柔軟に書けます
func ExampleComplexJoin(db *gorm.DB, userID uint) error {
	var records []struct {
		UserName      string
		RoleName      string
		AccessedAt    string
	}
	err := db.Table("users").
		Select("users.name AS user_name, roles.name AS role_name, access_histories.accessed_at").
		Joins("left join user_roles on user_roles.user_id = users.id").
		Joins("left join roles on roles.id = user_roles.role_id").
		Joins("left join access_histories on access_histories.user_id = users.id").
		Where("users.id = ?", userID).
		Scan(&records).Error

	if err == nil {
		for _, r := range records {
			fmt.Printf("User: %s, Role: %s, Accessed: %s\n", r.UserName, r.RoleName, r.AccessedAt)
		}
	}

	return err
}
