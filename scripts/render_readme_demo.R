args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: render_readme_demo.R signals.csv frames_dir frame_count scenario")

signals <- read.csv(args[[1]], check.names = FALSE)
frame_dir <- args[[2]]
frame_count <- as.integer(args[[3]])
scenario <- toupper(args[[4]])
time_s <- signals$time
channels <- c("ecg", "eda", "respiration", "accel_x")
labels <- c("ECG", "EDA", "RESPIRATION", "ACCELERATION · X AXIS")
colors <- c("#38bdf8", "#34d399", "#fbbf24", "#c084fc")
duration <- max(time_s)

dir.create(frame_dir, recursive = TRUE, showWarnings = FALSE)
for (frame_index in seq_len(frame_count)) {
  progress <- (frame_index - 1) / (frame_count - 1)
  cursor <- progress * duration
  png(file.path(frame_dir, sprintf("frame-%03d.png", frame_index)),
      width = 1280, height = 720, bg = "#090d12", res = 120)
  par(mfrow = c(4, 1), mar = c(1.3, 4.2, 1.8, 1.2), oma = c(4.4, 0.2, 6.0, 0.2),
      bg = "#090d12", fg = "#94a3b8", col.axis = "#94a3b8", col.lab = "#94a3b8",
      family = "sans", xaxs = "i", yaxs = "i")

  for (i in seq_along(channels)) {
    y <- signals[[channels[[i]]]]
    plot(time_s, y, type = "n", xlim = c(0, duration), ylim = c(-1.15, 1.15),
         axes = FALSE, xlab = "", ylab = "")
    rect(0, -1.15, duration, 1.15, col = "#0e151d", border = "#263341")
    for (tick in seq(0, duration, by = 1)) abline(v = tick, col = "#1b2835", lwd = 0.8)
    for (tick in c(-1, 0, 1)) abline(h = tick, col = "#1b2835", lwd = 0.8)
    lines(time_s, y, col = colors[[i]], lwd = 1.5)
    abline(v = cursor, col = "#f8fafc", lwd = 1.1, lty = 2)
    nearest <- which.min(abs(time_s - cursor))
    points(time_s[[nearest]], y[[nearest]], pch = 16, cex = 0.9, col = "#f8fafc")
    axis(2, at = c(-1, 0, 1), labels = c("−1", "0", "+1"), las = 1,
         col = "#334155", col.ticks = "#334155", cex.axis = 0.72)
    if (i == length(channels)) {
      axis(1, at = seq(0, duration, by = 2), labels = seq(0, duration, by = 2),
           col = "#334155", col.ticks = "#334155", cex.axis = 0.72)
      mtext("time (seconds)", side = 1, line = 2.1, col = "#94a3b8", cex = 0.7)
    }
    mtext(labels[[i]], side = 3, line = 0.35, adj = 0, col = colors[[i]],
          cex = 0.78, font = 2)
    mtext("normalized for display", side = 4, line = 0.2, col = "#64748b", cex = 0.55)
  }
  mtext("GUARDIAN DRIVE  /  SENSOR REPLAY", side = 3, outer = TRUE, line = 4.0,
        adj = 0, col = "#e2e8f0", cex = 1.3, font = 2)
  mtext(paste("DETERMINISTIC SIMULATOR", scenario, "SCENARIO"), side = 3, outer = TRUE,
        line = 2.2, adj = 0, col = "#38bdf8", cex = 0.8, font = 2)
  mtext("SYNTHETIC INPUT  ·  NO REAL PARTICIPANT DATA  ·  NO MODEL PREDICTIONS",
        side = 1, outer = TRUE, line = 1.7, adj = 0, col = "#94a3b8", cex = 0.72)
  mtext(sprintf("%04.1f / %04.1f s", cursor, duration), side = 1, outer = TRUE,
        line = 0.2, adj = 0, col = "#e2e8f0", cex = 0.8, font = 2)
  dev.off()
}
