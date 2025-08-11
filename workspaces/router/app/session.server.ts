import { createSessionStorage, type Cookie, type FlashSessionData, type SessionData, type SessionStorage } from "react-router";

interface SessionStorageConfig {
  cookie: Cookie;
  host: string;
  port: number;
}

export function createDatabaseSessionStorage({
  cookie,
  host,
  port,
}:SessionStorageConfig) {
  // Configure your database client...
  const db = createDatabaseClient(host, port);

  return createSessionStorage({
    cookie,
    async createData(data, expires) {
      // `expires` is a Date after which the data should be considered
      // invalid. You could use it to invalidate the data somehow or
      // automatically purge this record from your database.
      const id = await db.insert(data,expires);
      return id;
    },
    async readData(id) {
      return (await db.select(id)) || null;
    },
    async updateData(id, data, expires) {
      await db.update(id, data,expires);
    },
    async deleteData(id) {
      await db.delete(id);
    },
  });
}


