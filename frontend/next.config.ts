import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `next dev` otherwise writes its own AGENTS.md and CLAUDE.md into this
  // directory, shadowing the documentation kept at the repository root.
  agentRules: false,
};

export default nextConfig;
