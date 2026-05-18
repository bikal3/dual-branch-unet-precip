# Portfolio Website Design — Precipitation Downsampling

**Date:** 2026-05-18  
**Status:** Approved

---

## Goal

Build a static portfolio website that showcases the precipitation downsampling project to potential employers and collaborators. The site presents the problem, approach, and results, with an interactive Leaflet map demo comparing IMERG input vs. model prediction.

---

## Tech Stack

| Concern | Choice |
|---|---|
| Framework | Next.js 14 (`output: 'export'`) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Map | react-leaflet + leaflet |
| Deploy | gh-pages → GitHub Pages |

**Hosting:** `https://<username>.github.io/precipitation-downsampling/`  
No backend, no API, no database — fully static.

---

## Site Structure

Single-page app with sticky top nav linking to six sections:

| # | Section | Content |
|---|---|---|
| 1 | Hero | Project title, one-sentence summary, GitHub link |
| 2 | Problem | Why downscaling matters, what IMERG is, Big Island context |
| 3 | Approach | Inputs (IMERG, GOES, DEM), Dual-Branch U-Net architecture diagram, loss function |
| 4 | Results | Metrics table (best val loss 0.0361, test loss 0.0332) + loss curve image |
| 5 | Demo | Leaflet map with day-picker and IMERG↔Prediction opacity slider |
| 6 | About | Author (Bikal), course (ADLEO), team members |

---

## File Layout

```
website/
  pages/
    index.tsx                  # single page, imports all sections
  components/
    Hero.tsx
    Problem.tsx
    Approach.tsx
    Results.tsx
    Demo.tsx                   # Leaflet map + day-picker + slider
    About.tsx
    NavBar.tsx
  public/
    samples/
      YYYY-MM-DD/
        imerg.png              # IMERG input resampled to Big Island bbox
        pred.png               # model prediction at 250 m
    loss_curve.png             # copied from docs/loss_curve.png
  next.config.js               # basePath + assetPrefix for GitHub Pages
  tailwind.config.ts
  package.json
```

---

## Demo Component Design

**Map:** `react-leaflet` centered on Big Island bounding box `lon [-156.07, -154.799], lat [18.89, 20.277]`. OpenStreetMap basemap tiles (free, no API key).

**Day-picker:** Button group showing 3–5 pre-exported sample days from the Apr–Jun 2021 test set (e.g. "Apr 15", "May 3", "Jun 22"). Selecting a day loads that day's image pair.

**Layers:**
- Bottom: `imerg.png` — ImageOverlay pinned to Big Island bbox (shows coarse ~10 km blobs)
- Top: `pred.png` — ImageOverlay pinned to same bbox (shows sharp 250 m detail)

**Slider:** Labeled "IMERG ←→ Prediction", controls CSS opacity of the prediction layer (0 = full IMERG, 1 = full prediction). Makes the sharpening effect immediately visible.

**Legend:** Small color scale panel showing rainfall range in mm/day.

### Sample Images Required (pre-build)

For each of 3–5 selected test days, export from the trained model:
- `imerg.png` — IMERG input upsampled/resampled to Big Island extent, same color scale
- `pred.png` — model prediction output, same color scale and extent

Both images must share the same bounding box (`[-156.07, 18.89, -154.799, 20.277]`) so Leaflet overlays them correctly.

Place at: `public/samples/YYYY-MM-DD/imerg.png` and `public/samples/YYYY-MM-DD/pred.png`.

---

## Deployment

```json
// package.json scripts
"build": "next build",
"deploy": "next build && gh-pages -d out"
```

```js
// next.config.js
const nextConfig = {
  output: 'export',
  basePath: '/precipitation-downsampling',
  assetPrefix: '/precipitation-downsampling/',
}
```

Deploy command: `npm run deploy` — builds static files and pushes `out/` to the `gh-pages` branch.

---

## Out of Scope

- Live inference (model does not run in the browser)
- Multiple models (only Dual-Branch U-Net showcased)
- CMS or editable content
- Authentication or user accounts
