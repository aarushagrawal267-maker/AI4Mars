import { BookOpen, ExternalLink } from 'lucide-react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Callout, ClassSwatch, PageHeader } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import { archLabel, CLASS_KEYS, CLASS_LABEL, ratio } from '../utils/format';

const SECTIONS = [
  ['overview', 'Project overview'],
  ['architecture', 'Technical architecture'],
  ['dataset', 'Dataset'],
  ['models', 'Model architectures'],
  ['training', 'Training methodology'],
  ['evaluation', 'Evaluation methodology'],
  ['results', 'Current results'],
  ['navigation', 'Navigation assumptions'],
  ['limitations', 'Known limitations'],
  ['references', 'References'],
] as const;

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 border-b border-line/60 pb-10 pt-2 last:border-0">
      <h2 className="mb-4 text-xl font-semibold text-ink">{title}</h2>
      <div className="space-y-3 text-sm leading-relaxed text-ink-2 [&_b]:font-medium [&_b]:text-ink [&_code]:rounded [&_code]:bg-sunken [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[12px] [&_code]:text-ink [&_li]:ml-5 [&_li]:list-disc [&_li]:pl-1">
        {children}
      </div>
    </section>
  );
}

function Box({ title, items, accent }: { title: string; items: string[]; accent?: boolean }) {
  return (
    <div className={`rounded-lg border p-3 ${accent ? 'border-accent/50 bg-accent/[0.05]' : 'border-line bg-sunken/60'}`}>
      <div className="mb-1.5 font-mono text-2xs uppercase tracking-wider text-ink-3">{title}</div>
      <ul className="space-y-0.5 text-xs text-ink-2">{items.map((i) => <li key={i} className="!ml-0 !list-none !pl-0">{i}</li>)}</ul>
    </div>
  );
}

function LiveResults() {
  const { connection } = useApp();
  const ev = useRemote(() => api.evaluations(), [connection]);
  const tests = (ev.data?.evaluations ?? []).filter((e) => e.split === 'test' && !e.is_fixture);
  if (!tests.length) {
    return <Callout tone="info">No test-set evaluation has been run on this machine yet, so there are no results to report. Train and evaluate models to populate this section; nothing is hardcoded.</Callout>;
  }
  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="table-clean">
        <thead><tr><th>Model</th><th>Test labels</th><th className="text-right">Pixel acc.</th><th className="text-right">mIoU</th>{CLASS_KEYS.map((k) => <th key={k} className="text-right">{CLASS_LABEL[k]} IoU</th>)}</tr></thead>
        <tbody>
          {tests.map((e) => (
            <tr key={e.id}>
              <td><span className="chip mr-2">{archLabel(e.arch)}</span>{e.experiment_name}</td>
              <td className="num">{e.label_variant} · {e.images}</td>
              <td className="num text-right">{ratio(e.metrics.pixel_accuracy, 1)}</td>
              <td className="num text-right text-accent">{ratio(e.metrics.mean_iou, 1)}</td>
              {CLASS_KEYS.map((k) => <td key={k} className="num text-right">{ratio(e.metrics.per_class[k].iou, 1)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const A = ({ href, children }: { href: string; children: ReactNode }) => (
  <a href={href} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 text-accent hover:underline">{children}<ExternalLink className="h-3 w-3" /></a>
);

export default function Documentation() {
  return (
    <div>
      <PageHeader eyebrow="09 · reference" title="Documentation"
        description="How the system works, what it assumes, and where it falls short. The repository's docs/ folder contains the long-form versions of these sections." />
      <div className="grid gap-10 xl:grid-cols-[220px_1fr]">
        <nav className="hidden xl:block" aria-label="Documentation sections">
          <div className="sticky top-20 space-y-0.5">
            <div className="eyebrow mb-2 flex items-center gap-1.5"><BookOpen className="h-3 w-3" /> contents</div>
            {SECTIONS.map(([id, label], i) => (
              <a key={id} href={`#${id}`} className="flex gap-2 rounded-md px-2 py-1.5 text-xs text-ink-2 hover:bg-raised hover:text-ink">
                <span className="num text-ink-3">{String(i + 1).padStart(2, '0')}</span>{label}
              </a>
            ))}
          </div>
        </nav>
        <article className="min-w-0 max-w-4xl">
          <Section id="overview" title="Project overview">
            <p>
              <b>Mars Terrain Intelligence</b> is an independent student research and engineering project. It trains semantic segmentation
              models that classify every pixel of a Mars rover image as <b>soil</b>, <b>bedrock</b>, <b>sand</b> or <b>big rock</b>, evaluates them
              on expert-labelled held-out images, and uses the predictions in an experimental terrain-aware path-planning simulation.
            </p>
            <p>
              It is inspired by NASA JPL&apos;s AI4Mars dataset and research on terrain-aware autonomous driving. It is <b>not</b> an official NASA system,
              is not endorsed by NASA or JPL, and has never been deployed on, or connected to, a rover.
            </p>
            <div className="flex flex-wrap gap-4 pt-1">{CLASS_KEYS.map((k, i) => <span key={k} className="flex items-center gap-2 text-xs"><ClassSwatch k={k} /> <code>{i}</code> {CLASS_LABEL[k]}</span>)}<span className="text-xs"><code>255</code> unlabelled (ignored)</span></div>
          </Section>

          <Section id="architecture" title="Technical architecture">
            <div className="grid gap-3 md:grid-cols-3">
              <Box title="frontend/ · React + TypeScript" items={['Vite, Tailwind CSS, Recharts', '10 pages, React Router', 'Polls the REST API; no data of its own']} />
              <Box title="backend/ · FastAPI" accent items={['REST API + OpenAPI docs at :8000/docs', 'SQLite via SQLAlchemy (experiments, evaluations, analyses, settings)', 'Background jobs; A* planner']} />
              <Box title="ml/ · PyTorch" items={['Dataset acquisition, validation, caching', 'U-Net, DeepLabV3, training loop', 'Metrics, inference, terrain statistics']} />
            </div>
            <p>
              Training runs in a <b>separate Python process</b> started by the API (<code>python -m ml.training.run</code>). It communicates only through
              files in <code>experiments/&lt;id&gt;/</code>: <code>status.json</code>, <code>history.jsonl</code>, <code>train.log</code> and a <code>control.json</code> the API writes
              to pause or stop. This keeps the web server responsive and survives out-of-memory crashes. Checkpoints (<code>checkpoints/&lt;id&gt;/best.pt</code> and
              <code> last.pt</code>) are written atomically after each completed epoch, so stopping a run never loses finished work.
            </p>
            <p>
              Security choices: users never supply file-system paths (imports accept only bare file names from <code>data/raw/imports</code>); ZIP members that would
              escape the destination are rejected; checkpoints load with <code>torch.load(weights_only=True)</code>; uploads are size-limited and decoded/verified before use.
            </p>
          </Section>

          <Section id="dataset" title="Dataset">
            <p>
              Source: <A href="https://zenodo.org/records/15995036">Zenodo record 15995036</A> — <i>AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars</i>, merged
              release v0.6 (Swan, Ono, Atha), CC-BY-4.0, one 16.2 GB archive. The archive&apos;s own <code>info.md</code> and <code>label_keys.json</code> define the NAV label encoding,
              and it was verified on real files before any loader was written: 8-bit grayscale PNG, 1024×1024, values 0–3 plus 255 for no label.
            </p>
            <ul>
              <li><b>Subset used:</b> Curiosity (MSL) NavCam — the most thoroughly cleaned part according to the dataset changelog, and the only one with an expert test set for NavCam.</li>
              <li><b>Train labels</b> are crowdsourced (Zooniverse), merged with at least 3 labellers and 2/3 per-pixel agreement. <b>Test labels</b> come from rover planners and scientists, in three agreement variants (min1/min2/min3).</li>
              <li><b>Splits:</b> the official 322-image test partition is used unchanged. The training labels are split 90/10 into train/validation <b>by rover position</b> (site + drive parsed from the filename), so near-duplicate frames from one location never straddle the two.</li>
              <li><b>Validation:</b> every pair is checked for missing files, unreadable images, size mismatches, undocumented label values and train/test overlap; fully unlabelled masks are excluded. See the <Link className="text-accent hover:underline" to="/dataset">Dataset Explorer</Link>.</li>
              <li><b>Acquisition:</b> a subset can be pulled directly out of the remote ZIP with HTTP range requests (CRC-checked), the full archive downloaded (MD5-checked), or a local ZIP (Zenodo or the <A href="https://www.kaggle.com/datasets/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars">Kaggle mirror</A>) imported.</li>
            </ul>
          </Section>

          <Section id="models" title="Model architectures">
            <p>
              <b>U-Net</b> (Ronneberger, Fischer &amp; Brox, 2015) is a symmetric encoder–decoder. Each encoder stage applies two 3×3 convolutions with batch
              normalisation and ReLU, then halves resolution; the decoder upsamples and <b>concatenates the matching encoder feature map</b> (skip connections) so fine
              spatial detail lost in downsampling is recovered. Our implementation is configurable (base channels, depth, bilinear vs transposed upsampling, dropout) and is trained from scratch.
            </p>
            <p>
              <b>DeepLabV3</b> (Chen et al., 2017) replaces aggressive downsampling with <b>atrous (dilated) convolutions</b> and adds <b>Atrous Spatial Pyramid Pooling</b>
              — parallel dilated convolutions plus image-level pooling — to capture context at several scales. We use Torchvision&apos;s implementation with a MobileNetV3-Large
              (default) or ResNet-50 backbone, initialised from Torchvision&apos;s COCO-trained segmentation weights, and replace the final 1×1 classifier with a new 4-class layer.
            </p>
            <p>Both models receive identical inputs: the grayscale NavCam frame replicated to three channels and normalised with ImageNet statistics.</p>
          </Section>

          <Section id="training" title="Training methodology">
            <ul>
              <li>Loss: cross-entropy with <code>ignore_index=255</code>, optionally plus soft Dice (default weight 0.5). Unlabelled pixels contribute nothing to either term.</li>
              <li>Optimiser: AdamW with cosine annealing (or reduce-on-plateau / constant); early stopping on validation mIoU; best checkpoint selected on validation mIoU only.</li>
              <li>Augmentation: horizontal flip, random scale-crop (60–100%), brightness/contrast jitter, mild noise. No vertical flips — the sky/horizon/near-field layout is meaningful.</li>
              <li>Images are resized once per resolution into a memory-mapped cache; labels use nearest-neighbour resizing so class ids are never blended.</li>
              <li>Before a run starts, its duration is <b>measured</b> by timing real optimisation steps on your hardware; runs estimated above two hours require explicit confirmation.</li>
              <li><b>Fair comparison:</b> same data split, resolution, batch size, epochs, optimiser, learning rate, loss, augmentation and seed — only the architecture differs. The test set is touched only for final evaluation.</li>
            </ul>
          </Section>

          <Section id="evaluation" title="Evaluation methodology">
            <p>
              Each held-out image is predicted at the model&apos;s input resolution; class probabilities are upsampled back to the label&apos;s native 1024×1024 resolution and compared
              with the original annotation. A 4×4 confusion matrix is accumulated over <b>labelled pixels only</b>. From it we compute pixel accuracy, per-class IoU, Dice,
              precision and recall, and their means over classes present in the ground truth. Undefined values (a class never predicted or never present) are reported as
              <b> n/a</b>, not zero.
            </p>
            <p>
              Inference time is the median single-image forward pass on the evaluation device, excluding the first warm-up image. Model size is reported as parameter count
              and checkpoint bytes. Confidence shown elsewhere is the maximum softmax probability; <b>its calibration has not been tested</b>, so it must not be read as the
              probability that a prediction is correct.
            </p>
          </Section>

          <Section id="results" title="Current results">
            <p>Live from this machine&apos;s database — test-set evaluations only:</p>
            <LiveResults />
          </Section>

          <Section id="navigation" title="Navigation simulation assumptions">
            <ul>
              <li>The predicted class map is downsampled to a grid by majority vote. A* (8-connected, octile heuristic scaled by the cheapest terrain cost, no corner cutting) finds the minimum-cost path; the heuristic is admissible, so the route is optimal for the chosen costs.</li>
              <li>Moving between neighbouring cells costs distance × the mean of their terrain costs. Costs, impassable classes and restricted cells are <b>user assumptions</b>, not validated mobility measurements. No class is assumed safe or unsafe by default.</li>
              <li>Cells are image-space, not metres. Perspective means a cell near the horizon covers far more ground than one near the rover.</li>
              <li>Terrain classification alone cannot determine rover safety: slope, rock height, sinkage, rover state and mission rules all matter. The simulator is not connected to any rover.</li>
            </ul>
          </Section>

          <Section id="limitations" title="Known limitations">
            <ul>
              <li><b>Class imbalance:</b> big rock is roughly 1% of labelled training pixels and under 0.2% of test pixels, so its scores are noisy and typically low.</li>
              <li><b>Label quality and bias:</b> training labels are crowdsourced and sparse (~40% of pixels unlabelled); terrain beyond ~30 m and the rover body are masked out. Models never learn those regions, yet inference still assigns them a class — in practice the trained models often label the rover&apos;s own hardware as bedrock and distant ridges as big rock. Those pixels should be ignored, and they also inflate the composition percentages of whole-image analyses.</li>
              <li><b>Domain:</b> trained on grayscale Curiosity NavCam images. Colour images, other rovers, other cameras or other terrain types are out of distribution; colour uploads are converted to grayscale.</li>
              <li><b>Compute:</b> CPU-only machines can train only small subsets at reduced resolution for a few epochs. Results from such runs are genuine but well below what full-dataset GPU training achieves.</li>
              <li><b>Single seed:</b> the baseline comparison uses one seed; differences of a few points may not be significant.</li>
              <li><b>Generalisation:</b> some test images come from rover positions also present in training data (the Dataset Explorer reports how many), which may flatter scores.</li>
            </ul>
          </Section>

          <Section id="references" title="References">
            <ol className="list-decimal space-y-2 pl-5 [&>li]:ml-0 [&>li]:list-decimal">
              <li>R. M. Swan, D. Atha, H. A. Leopold, M. Gildner, S. Oij, C. Chiu, M. Ono. &ldquo;AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars.&rdquo; CVPR Workshops, 2021.</li>
              <li>R. Swan, M. Ono, D. Atha. AI4Mars merged dataset v0.6. Zenodo, 2022. <A href="https://doi.org/10.5281/zenodo.15995036">doi:10.5281/zenodo.15995036</A></li>
              <li>O. Ronneberger, P. Fischer, T. Brox. &ldquo;U-Net: Convolutional Networks for Biomedical Image Segmentation.&rdquo; MICCAI, 2015.</li>
              <li>L.-C. Chen, G. Papandreou, F. Schroff, H. Adam. &ldquo;Rethinking Atrous Convolution for Semantic Image Segmentation.&rdquo; arXiv:1706.05587, 2017.</li>
              <li>A. Howard et al. &ldquo;Searching for MobileNetV3.&rdquo; ICCV, 2019.</li>
              <li>K. He, X. Zhang, S. Ren, J. Sun. &ldquo;Deep Residual Learning for Image Recognition.&rdquo; CVPR, 2016.</li>
              <li>I. Loshchilov, F. Hutter. &ldquo;Decoupled Weight Decay Regularization.&rdquo; ICLR, 2019.</li>
              <li>F. Milletari, N. Navab, S.-A. Ahmadi. &ldquo;V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.&rdquo; 3DV, 2016.</li>
              <li>P. E. Hart, N. J. Nilsson, B. Raphael. &ldquo;A Formal Basis for the Heuristic Determination of Minimum Cost Paths.&rdquo; IEEE Trans. Systems Science and Cybernetics, 1968.</li>
              <li>C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. &ldquo;On Calibration of Modern Neural Networks.&rdquo; ICML, 2017.</li>
              <li>AI4Mars on Zooniverse: <A href="https://www.zooniverse.org/projects/hiro-ono/ai4mars">zooniverse.org/projects/hiro-ono/ai4mars</A></li>
            </ol>
          </Section>
        </article>
      </div>
    </div>
  );
}
