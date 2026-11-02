import { defineConfig } from "astro/config";
import svelte from "@astrojs/svelte";

// Deployed to GitHub Pages at https://priyanshu-mishra.github.io/aegis
export default defineConfig({
  site: "https://priyanshu-mishra.github.io",
  base: "/aegis",
  devToolbar: { enabled: false },
  trailingSlash: "ignore",
  integrations: [svelte()],
});
