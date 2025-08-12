package db

import (
	"fmt"
	"gorm.io/gorm"
)

// CreateUser : ユーザ作成
func CreateUser(db *gorm.DB, user *User) error {
	return db.Create(user).Error
}

// UpdateUserName : ユーザ名更新
func UpdateUserName(db *gorm.DB, id uint, newName string) error {
	return db.Model(&User{}).Where("id = ?", id).Update("name", newName).Error
}

// GetUserWithPosts : ユーザとその投稿一覧取得（Preload版）
func GetUserWithPosts(db *gorm.DB, id uint) (User, error) {
	var u User
	err := db.Preload("Posts").First(&u, id).Error
	return u, err
}

// JoinUserPosts : Joinしてユーザ名と投稿タイトル取得
func JoinUserPosts(db *gorm.DB) ([]map[string]interface{}, error) {
	var results []map[string]interface{}
	err := db.Model(&Post{}).
		Select("users.name as user_name, posts.title as post_title").
		Joins("left join users on users.id = posts.user_id").
		Scan(&results).Error
	return results, err
}

// TransactionExample : トランザクションのリード・ライト例
func TransactionExample(db *gorm.DB, userID uint) error {
	return db.Transaction(func(tx *gorm.DB) error {
		// リード
		var count int64
		if err := tx.Model(&Post{}).Where("user_id = ?", userID).Count(&count).Error; err != nil {
			return err
		}
		fmt.Println("Post count:", count)

		// ライト
		newPost := Post{UserID: userID, Title: "New Post", Content: "Created in transaction."}
		if err := tx.Create(&newPost).Error; err != nil {
			return err
		}
		return nil
	})
}

// DeletePost : 投稿削除
func DeletePost(db *gorm.DB, postID uint) error {
	return db.Delete(&Post{}, postID).Error
}
