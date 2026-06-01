import express from "express";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import { request as httpRequest } from "http";
import { createServer as createViteServer } from "vite";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // 1. Spawning Python Background Microservice
  let pythonPath = "python";
  const possiblePaths = [
    path.join(process.cwd(), ".venv", "Scripts", "python.exe"), // Windows VirtualEnv
    path.join(process.cwd(), ".venv", "bin", "python"),        // Unix VirtualEnv
    path.join(process.cwd(), "venv", "Scripts", "python.exe"),
    path.join(process.cwd(), "venv", "bin", "python"),
  ];

  for (const p of possiblePaths) {
    if (fs.existsSync(p)) {
      pythonPath = p;
      break;
    }
  }

  console.log(`[*] Spawning backend model_server using: ${pythonPath}`);
  const pythonProcess = spawn(pythonPath, ["model_server.py"], {
    stdio: "inherit",
    shell: false
  });

  pythonProcess.on("error", (err) => {
    console.error("[-] Error launching Python model subprocess:", err);
  });

  // Handle clean terminations
  process.on("exit", () => pythonProcess.kill());
  process.on("SIGINT", () => {
    pythonProcess.kill();
    process.exit();
  });

  // 2. Proxy request pipeline to forward /api/* calls to FastAPI server on port 5000
  app.all("/api/*", (req, res) => {
    const options = {
      hostname: "127.0.0.1",
      port: 5000,
      path: req.originalUrl,
      method: req.method,
      headers: req.headers
    };

    const proxyReq = httpRequest(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res);
    });

    proxyReq.on("error", (err) => {
      console.warn("[!] Proxy forward failed ( FastAPI may still be loading ):", err.message);
      res.status(503).json({ error: "VoiceGuard Model API Server is starting, please retry." });
    });

    req.pipe(proxyReq);
  });

  // 3. Vite development server asset loading or production single-page hosting
  if (process.env.NODE_ENV !== "production") {
    console.log("[*] Mounting Vite middleware in development mode...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("[*] Serving production-ready React client assets...");
    const distPath = path.join(process.cwd(), "dist");
    if (!fs.existsSync(distPath)) {
      console.warn("[!] Warning: dist/ folder does not exist. Run 'npm run build' first.");
    }
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`================================================================`);
    console.log(`[+] VoiceGuard Hub operational URL: http://localhost:${PORT}`);
    console.log(`================================================================`);
  });
}

startServer().catch((e) => {
  console.error("[-] Server initialization crashed:", e);
});
