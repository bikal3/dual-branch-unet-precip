const ARCH_DIAGRAM = `
Input (B, 5, 32, 32)
        │
        ├─── Branch 1 (IMERG + GOES, 2 ch) ─── Branch 2 (Topo, 3 ch) ───┐
        │    ConvBlock(2→32) → Pool              ConvBlock(3→32) → Pool   │
        │    ConvBlock(32→64) → Pool             ConvBlock(32→64) → Pool  │
        │    ConvBlock(64→128) → Pool            ConvBlock(64→128) → Pool │
        └──────────── cat(dim=1) → (B, 256, 4, 4) ───────────────────────┘
                          ConvBlock(256→256)
                               │
                    ── Shared Decoder ──
              ConvTranspose2d(256→128) → ConvBlock(128→128)
              ConvTranspose2d(128→64)  → ConvBlock(64→64)
              ConvTranspose2d(64→32)   → ConvBlock(32→32)
                       Conv2d(32→1, 1×1)
                               │
                      Output (B, 1, 32, 32)
`.trim();

export default function Approach() {
  return (
    <section id="approach" className="bg-gray-50 py-20 px-6">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-3xl font-bold text-gray-900">Approach</h2>
        <p className="mt-4 text-gray-700 leading-relaxed">
          The model is a <strong>Dual-Branch U-Net</strong> that processes two
          input modalities through independent encoder branches before fusing
          them at a shared bottleneck.
        </p>

        <div className="mt-6 space-y-2 text-gray-700">
          <div className="flex gap-2">
            <span className="font-semibold w-24 shrink-0">Branch 1</span>
            <span>IMERG precipitation + GOES-17 cloud mask (2 channels)</span>
          </div>
          <div className="flex gap-2">
            <span className="font-semibold w-24 shrink-0">Branch 2</span>
            <span>DEM elevation, slope, aspect (3 channels)</span>
          </div>
        </div>

        <h3 className="mt-8 text-xl font-semibold text-gray-800">
          Architecture
        </h3>
        <pre data-testid="arch-diagram" className="mt-3 w-full overflow-x-auto rounded-md bg-gray-900 p-4 text-xs leading-relaxed text-green-300">
          {ARCH_DIAGRAM}
        </pre>

        <h3 className="mt-8 text-xl font-semibold text-gray-800">
          Loss Function
        </h3>
        <p className="mt-2 text-gray-700">
          <code className="rounded bg-gray-200 px-1.5 py-0.5 text-sm">
            Loss = λ₁ · MaskedMSE + λ₂ · TVLoss
          </code>
        </p>
        <p className="mt-2 text-gray-700 text-sm">
          MaskedMSE computes error only at rain gauge pixels (station mask).
          TVLoss penalises abrupt spatial gradients, encouraging smooth
          precipitation fields. λ₁&nbsp;=&nbsp;1.0, λ₂&nbsp;=&nbsp;0.1.
        </p>
      </div>
    </section>
  );
}
