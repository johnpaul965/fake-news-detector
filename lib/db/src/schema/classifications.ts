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
