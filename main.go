package main

import (
	"database/sql"
	"fmt"
	"log"
	_ "github.com/lib/pq"
    "src/pkg"
)

func main() {
	db, err := sql.Open("postgres", "postgres://user:password@localhost:5432/dbname?sslmode=disable")
	if err != nil {
		log.Fatalf("DB接続失敗: %v", err)
	}
	defer db.Close()

    ms:=pkg.Message()
	fmt.Println("✅ DB接続OK",ms)

}
