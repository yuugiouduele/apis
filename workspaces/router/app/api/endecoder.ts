import fs from "fs";
import path from "path";

/**
 * ファイルをBase64文字列に変換
 * @param filePath 変換したいファイルのパス
 * @returns Base64エンコードされた文字列
 */
function encodeFileToBase64(filePath: string): string {
  const fileBuffer = fs.readFileSync(filePath);
  return fileBuffer.toString("base64");
}

/**
 * Base64文字列をファイルとして保存
 * @param base64String Base64文字列
 * @param outputPath ファイル保存先パス
 */
function decodeBase64ToFile(base64String: string, outputPath: string): void {
  const fileBuffer = Buffer.from(base64String, "base64");
  fs.writeFileSync(outputPath, fileBuffer);
}

// --- 使用例 ---
(async () => {
  const inputFile = path.join(__dirname, "sample.png");      // 元ファイル
  const encodedFile = path.join(__dirname, "encoded.txt");   // Base64文字列の保存先
  const decodedFile = path.join(__dirname, "decoded.png");   // 復元ファイル

  // 1. ファイル → Base64
  const base64Data = encodeFileToBase64(inputFile);
  fs.writeFileSync(encodedFile, base64Data, { encoding: "utf-8" });
  console.log("✅ Base64エンコード完了");

  // 2. Base64 → ファイル
  const base64StringFromFile = fs.readFileSync(encodedFile, "utf-8");
  decodeBase64ToFile(base64StringFromFile, decodedFile);
  console.log("✅ Base64デコード完了");
})();
