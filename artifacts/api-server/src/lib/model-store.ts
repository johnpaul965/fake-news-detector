import { trainAllModels, type TrainedModels } from "./model-trainer.js";
import { logger } from "./logger.js";

let store: TrainedModels | null = null;

export async function initModel(): Promise<void> {
  logger.info("Training Random Forest models (3 configurations, 10-fold CV)...");
  const start = Date.now();
  store = trainAllModels();
  logger.info(
    { elapsed: `${Date.now() - start}ms`, dataSource: store.dataSource },
    "Models trained and ready",
  );
}

export function getModelStore(): TrainedModels {
  if (!store) throw new Error("Model store not initialized");
  return store;
}
