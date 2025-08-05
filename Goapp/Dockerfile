FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN go mod tidy
RUN go build -o main .

FROM debian:bullseye-slim
WORKDIR /app
COPY --from=builder /app/main .
COPY --from=builder /app/.env ./  # .envがあれば
EXPOSE 8080
CMD ["./main"]
