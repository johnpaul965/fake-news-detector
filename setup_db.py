"""
Run this once to create the missing database schema files locally.
Usage: python setup_db.py
"""
import os

os.makedirs("lib/db/src/schema", exist_ok=True)

files = {
    "lib/db/src/schema/classifications.ts": '''\
import { pgTable, serial, text, real, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const classificationsTable = pgTable("classifications", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  bodyPreview: text("body_preview").notNull(),
  prediction: text("prediction").notNull(),
  confidence: real("confidence").notNull(),
  featuresJson: jsonb("features_json").notNull(),
  experimentsJson: jsonb("experiments_json").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertClassificationSchema = createInsertSchema(classificationsTable).omit({ id: true, createdAt: true });
export type InsertClassification = z.infer<typeof insertClassificationSchema>;
export type Classification = typeof classificationsTable.$inferSelect;
''',
    "lib/db/src/schema/index.ts": '''\
export * from "./classifications";
''',
    "lib/db/src/index.ts": '''\
import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "./schema";

const { Pool } = pg;

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL must be set. Did you forget to provision a database?");
}

export const pool = new Pool({ connectionString: process.env.DATABASE_URL });
export const db = drizzle(pool, { schema });

export * from "./schema";
''',
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nDone! Now run:")
print('  $env:DATABASE_URL = "postgresql://postgres:YourPassword@localhost:5432/fakenews"')
print("  pnpm --filter @workspace/db run push")
