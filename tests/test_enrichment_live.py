"""
Live API test for batched enrichment with synthetic chunks.

Tests the soul+constitution prompt against real OpenRouter API with
varied chunk types to validate structured output reliability.

Run with: pytest tests/test_enrichment_live.py -v -s
"""

import asyncio
import os
import sys

import pytest

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from code_rag.enrichment import (
    _call_llm_batch,
    enrich_chunks,
    get_embedding_text,
)
from code_rag.models import Chunk
from code_rag.providers import OpenRouterLLM

# This module hits the real OpenRouter API — excluded from CI via `-m "not live"`.
pytestmark = pytest.mark.live

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "minimax/minimax-m2.7"
llm = OpenRouterLLM(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Synthetic chunks — realistic samples from different languages/types
# ---------------------------------------------------------------------------

SYNTHETIC_CHUNKS = [
    # 1. Python class with methods (Manim-like)
    Chunk(
        text="""\
class VMobject(Mobject):
    \"\"\"Vectorized Mobject: base class for all vector graphics objects.\"\"\"

    CONFIG = {
        "fill_opacity": 0.0,
        "stroke_opacity": 1.0,
        "stroke_width": 4,
        "stroke_color": WHITE,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = np.zeros((0, 3))

    def get_fill_opacity(self):
        return self.fill_opacity

    def set_fill(self, color=None, opacity=None, family=True):
        if color is not None:
            self.fill_color = Color(color)
        if opacity is not None:
            self.fill_opacity = opacity
        if family:
            for submob in self.submobjects:
                submob.set_fill(color, opacity, family)
        return self""",
        file_name="manimlib/mobject/types/vectorized_mobject.py",
        section_title="VMobject",
        start_line=45,
        end_line=72,
        chunk_type="code",
        language="python",
        symbol_name="VMobject",
        symbol_type="class",
    ),
    # 2. GLSL fragment shader
    Chunk(
        text="""\
#version 330

uniform vec3 light_source_position;
uniform float gloss;
uniform float shadow;

in vec3 v_point;
in vec3 v_normal;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_source_position - v_point);
    float diffuse = max(dot(normal, light_dir), 0.0);
    float ambient = 0.2;
    float lighting = ambient + (1.0 - ambient) * diffuse;
    frag_color = vec4(v_color.rgb * lighting, v_color.a);
}""",
        file_name="manimlib/shaders/surface/frag.glsl",
        section_title="surface_frag",
        start_line=1,
        end_line=22,
        chunk_type="code",
        language="glsl",
        symbol_name="main",
        symbol_type="function",
    ),
    # 3. Python animation function
    Chunk(
        text="""\
def interpolate_mobject(self, alpha):
    \"\"\"
    Interpolate the mobject between its starting and target states.

    alpha ranges from 0 (start) to 1 (end). The rate_func is applied
    before interpolation to control easing.
    \"\"\"
    families = list(zip(
        self.mobject.get_family(),
        self.starting_mobject.get_family(),
        self.target_mobject.get_family(),
    ))
    for sm, start, target in families:
        self.interpolate_submobject(sm, start, target, alpha)""",
        file_name="manimlib/animation/animation.py",
        section_title="interpolate_mobject",
        start_line=120,
        end_line=135,
        chunk_type="code",
        language="python",
        symbol_name="interpolate_mobject",
        symbol_type="method",
    ),
    # 4. Markdown documentation
    Chunk(
        text="""\
## Rate Functions

Rate functions control the easing of animations. They map the linear
progression `alpha` (0 to 1) to a transformed value.

### Built-in rate functions

- `smooth` — zero derivative at endpoints, natural feel
- `linear` — constant rate, no easing
- `rush_into` — fast start, slow end
- `rush_from` — slow start, fast end
- `there_and_back` — forward then backward

### Custom rate functions

You can pass any callable that maps [0, 1] -> [0, 1]:

```python
scene.play(Transform(a, b), rate_func=lambda t: t**2)
```""",
        file_name="docs/rate_functions.md",
        section_title="Rate Functions",
        start_line=1,
        end_line=20,
        chunk_type="markdown",
        language=None,
    ),
    # 5. Short Python utility (edge case — minimal context)
    Chunk(
        text="""\
def smooth(t, inflection=10.0):
    error = sigmoid(-inflection / 2)
    return np.clip(
        (sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error),
        0, 1,
    )""",
        file_name="manimlib/utils/rate_functions.py",
        section_title="smooth",
        start_line=15,
        end_line=21,
        chunk_type="code",
        language="python",
        symbol_name="smooth",
        symbol_type="function",
    ),
]


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def test_single_batch():
    """Test a single batch of 5 diverse chunks."""
    print_separator("TEST 1: Single batch of 5 diverse chunks")

    results = await _call_llm_batch(SYNTHETIC_CHUNKS, llm, MODEL)

    success = 0
    for i, ((summary, questions), chunk) in enumerate(zip(results, SYNTHETIC_CHUNKS, strict=True)):
        print(f"--- Chunk {i + 1}: {chunk.file_name} ({chunk.chunk_type}) ---")
        if summary:
            print(f"  Summary: {summary}")
            print(f"  Questions: {questions}")
            success += 1
        else:
            print("  FAILED: No enrichment returned")
        print()

    print(f"Result: {success}/{len(SYNTHETIC_CHUNKS)} chunks enriched")
    return success == len(SYNTHETIC_CHUNKS)


async def test_small_batches():
    """Test with batch_size=2 to verify smaller batches work."""
    print_separator("TEST 2: Small batches (batch_size=2)")

    # Use just 4 chunks
    chunks = [
        Chunk(
            text="def add_updater(self, update_function, index=None, call_updater=True):\n"
            "    if index is None:\n"
            "        self.updaters.append(update_function)\n"
            "    else:\n"
            "        self.updaters.insert(index, update_function)\n"
            "    if call_updater:\n"
            "        update_function(self)\n"
            "    return self\n",
            file_name="manimlib/mobject/mobject.py",
            section_title="add_updater",
            start_line=200,
            end_line=208,
            chunk_type="code",
            language="python",
            symbol_name="add_updater",
            symbol_type="method",
        ),
        Chunk(
            text="precision mediump float;\n"
            "in vec2 uv_coords;\n"
            "uniform sampler2D Texture;\n"
            "out vec4 frag_color;\n\n"
            "void main() {\n"
            "    frag_color = texture(Texture, uv_coords);\n"
            "}\n",
            file_name="manimlib/shaders/image/frag.glsl",
            section_title="image_frag",
            start_line=1,
            end_line=8,
            chunk_type="code",
            language="glsl",
            symbol_name="main",
            symbol_type="function",
        ),
        Chunk(
            text="## Coordinate Systems\n\n"
            "Manim provides several coordinate system classes:\n\n"
            "- `Axes` — standard 2D Cartesian axes\n"
            "- `ThreeDAxes` — 3D Cartesian axes\n"
            "- `NumberPlane` — full 2D grid\n"
            "- `ComplexPlane` — complex number plane\n\n"
            "Use `coords_to_point(x, y)` (or `c2p`) to convert.\n",
            file_name="docs/coordinates.md",
            section_title="Coordinate Systems",
            start_line=1,
            end_line=10,
            chunk_type="markdown",
            language=None,
        ),
        Chunk(
            text='[tool.poetry]\nname = "manim"\nversion = "0.18.0"\n'
            'description = "Animation engine for explanatory math videos"\n'
            'authors = ["3Blue1Brown"]\n'
            'license = "MIT"\n',
            file_name="pyproject.toml",
            section_title="pyproject.toml",
            start_line=1,
            end_line=6,
            chunk_type="plaintext",
            language=None,
        ),
    ]

    result = await enrich_chunks(chunks, llm, MODEL, batch_size=2, max_concurrent=1)

    success = 0
    for i, chunk in enumerate(result):
        print(f"--- Chunk {i + 1}: {chunk.file_name} ---")
        if chunk.summary:
            print(f"  Summary: {chunk.summary}")
            print(f"  Questions: {chunk.hypothetical_questions}")
            success += 1
        else:
            print("  FAILED: No enrichment")
        print()

    print(f"Result: {success}/{len(chunks)} chunks enriched")
    return success == len(chunks)


async def test_embedding_text_integration():
    """Verify get_embedding_text produces correct output with enrichment."""
    print_separator("TEST 3: Embedding text integration")

    chunks = [SYNTHETIC_CHUNKS[0]]  # Just the VMobject class
    result = await _call_llm_batch(chunks, llm, MODEL)

    summary, questions = result[0]
    if summary:
        chunk = SYNTHETIC_CHUNKS[0]
        chunk.summary = summary
        chunk.hypothetical_questions = questions

        embed_text = get_embedding_text(chunk)
        print(f"Original text length: {len(chunk.text)} chars")
        print(f"Enriched text length: {len(embed_text)} chars")
        print(f"Added: {len(embed_text) - len(chunk.text)} chars of enrichment")
        print("\n--- Enrichment portion ---")
        enrichment = embed_text[len(chunk.text) :]
        print(enrichment)
        return True
    else:
        print("FAILED: No enrichment returned")
        return False


async def test_consistency():
    """Run the same batch 3 times to check output consistency."""
    print_separator("TEST 4: Consistency check (3 runs on same input)")

    chunk = Chunk(
        text="def bezier(points):\n"
        "    n = len(points) - 1\n"
        "    return lambda t: sum(\n"
        "        comb(n, k) * (1 - t)**(n - k) * t**k * p\n"
        "        for k, p in enumerate(points)\n"
        "    )\n",
        file_name="manimlib/utils/bezier.py",
        section_title="bezier",
        start_line=10,
        end_line=16,
        chunk_type="code",
        language="python",
        symbol_name="bezier",
        symbol_type="function",
    )

    summaries = []
    for run in range(3):
        results = await _call_llm_batch([chunk], llm, MODEL, max_retries=1)
        summary, questions = results[0]
        if summary:
            summaries.append(summary)
            print(f"Run {run + 1}: {summary}")
        else:
            print(f"Run {run + 1}: FAILED")
        # Small delay between runs for rate limiting
        await asyncio.sleep(2)

    if len(summaries) == 3:
        print("\nAll 3 runs produced valid output")
        return True
    else:
        print(f"\n{3 - len(summaries)} runs failed")
        return False


async def test_batch_8():
    """Test a single batch of 8 chunks — stress test for larger batches."""
    print_separator("TEST 5: Single batch of 8 diverse chunks")

    extra_chunks = [
        # 6. Rust-like systems code
        Chunk(
            text="""\
impl Scene {
    pub fn play(&mut self, animation: Box<dyn Animation>) {
        let run_time = animation.get_run_time();
        let num_frames = (run_time * self.camera.fps as f64) as usize;
        for frame in 0..num_frames {
            let alpha = frame as f64 / num_frames as f64;
            let t = (animation.rate_func)(alpha);
            animation.interpolate(t);
            self.camera.capture_frame(&self.mobjects);
        }
        animation.finish();
    }

    pub fn wait(&mut self, duration: f64) {
        let num_frames = (duration * self.camera.fps as f64) as usize;
        for _ in 0..num_frames {
            self.camera.capture_frame(&self.mobjects);
        }
    }
}""",
            file_name="src/scene/scene.rs",
            section_title="Scene",
            start_line=50,
            end_line=70,
            chunk_type="code",
            language="rust",
            symbol_name="Scene",
            symbol_type="class",
        ),
        # 7. GLSL vertex shader
        Chunk(
            text="""\
#version 330

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;
uniform float is_fixed_in_frame;

in vec3 point;
in vec3 base_normal;
in vec4 stroke_rgba;
in float stroke_width;
in float joint_angle;

out vec3 v_point;
out vec3 v_normal;
out vec4 v_color;
out float v_stroke_width;

void main() {
    v_point = (model_view_matrix * vec4(point, 1.0)).xyz;
    v_normal = normalize(mat3(model_view_matrix) * base_normal);
    v_color = stroke_rgba;
    v_stroke_width = stroke_width;
    gl_Position = projection_matrix * vec4(v_point, 1.0);
}""",
            file_name="manimlib/shaders/quadratic_bezier/stroke/vert.glsl",
            section_title="bezier_stroke_vert",
            start_line=1,
            end_line=25,
            chunk_type="code",
            language="glsl",
            symbol_name="main",
            symbol_type="function",
        ),
        # 8. Python complex math utility
        Chunk(
            text="""\
def get_smooth_quadratic_bezier_handle_points(points):
    \"\"\"
    Given a set of anchors, compute smooth handle points so that
    the resulting quadratic bezier curve passes through each anchor
    with continuous first derivatives.

    Returns an array of handle points interleaved between anchors.
    \"\"\"
    num_handles = len(points) - 1
    dim = points.shape[1]
    # Build tridiagonal system
    diag = np.ones(num_handles) * 4
    diag[0] = 2
    diag[-1] = 7
    upper = np.ones(num_handles - 1)
    lower = np.ones(num_handles - 1)
    lower[-1] = 2
    b = np.zeros((num_handles, dim))
    for i in range(num_handles):
        if i == 0:
            b[i] = points[0] + 2 * points[1]
        elif i == num_handles - 1:
            b[i] = 8 * points[-2] + points[-1]
        else:
            b[i] = 4 * points[i] + 2 * points[i + 1]
    from scipy.linalg import solve_banded
    ab = np.zeros((3, num_handles))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return solve_banded((1, 1), ab, b)""",
            file_name="manimlib/utils/bezier.py",
            section_title="get_smooth_quadratic_bezier_handle_points",
            start_line=45,
            end_line=78,
            chunk_type="code",
            language="python",
            symbol_name="get_smooth_quadratic_bezier_handle_points",
            symbol_type="function",
        ),
    ]

    all_8 = SYNTHETIC_CHUNKS + extra_chunks
    assert len(all_8) == 8

    results = await _call_llm_batch(all_8, llm, MODEL)

    success = 0
    for i, ((summary, questions), chunk) in enumerate(zip(results, all_8, strict=True)):
        print(f"--- Chunk {i + 1}: {chunk.file_name} ({chunk.language or chunk.chunk_type}) ---")
        if summary:
            print(f"  Summary: {summary}")
            print(f"  Questions: {questions}")
            success += 1
        else:
            print("  FAILED: No enrichment returned")
        print()

    print(f"Result: {success}/{len(all_8)} chunks enriched")
    return success == len(all_8)


async def main():
    if not API_KEY:
        print("ERROR: No OPENROUTER_API_KEY found in .env")
        return

    print(f"Using model: {MODEL}")
    print(f"API key: {API_KEY[:20]}...")

    results = {}

    results["batch_8"] = await test_batch_8()

    print_separator("FINAL RESULTS")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")


if __name__ == "__main__":
    asyncio.run(main())
