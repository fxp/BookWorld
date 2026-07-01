import os
import re
import json
import csv


class StoryProcessor:
    """
    Extracts world info, characters, and locations from a story text using LLM,
    then writes all data files required by BookWorld and returns a preset filename.
    """

    def __init__(self, llm_name: str, progress_callback=None):
        from bw_utils import get_models
        self.llm = get_models(llm_name)
        self._progress = progress_callback or (lambda step, pct: None)

    # ------------------------------------------------------------------ helpers

    def _call_llm(self, prompt: str) -> str:
        self.llm.initialize_message()
        self.llm.system_message(
            "You are a literary analysis assistant. "
            "Return ONLY valid JSON with no markdown fences."
        )
        self.llm.user_message(prompt)
        return self.llm.get_response(temperature=0.3)

    def _parse_json(self, text: str):
        # Strip optional ```json … ``` fences the model may add
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text).strip()
        return json.loads(text)

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]", "_", text)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "CustomWorld"

    # ---------------------------------------------------------------- extraction

    def _extract_world_info(self, text: str, title: str, language: str) -> dict:
        lang_label = "Chinese" if language == "zh" else "English"
        prompt = f"""Analyze the following novel excerpt and extract world background.

Novel title: {title}
Output language: {lang_label}

Return a JSON object with these exact keys:
{{
  "world_slug": "Short_English_Identifier_for_filenames",
  "world_name": "World name in {lang_label}",
  "description": "One-sentence world summary in {lang_label}",
  "detail": "3-5 sentence world description covering time period, society, culture in {lang_label}"
}}

Novel excerpt:
{text[:5000]}"""

        return self._parse_json(self._call_llm(prompt))

    def _extract_characters(self, text: str, language: str, lang_suffix: str) -> list:
        lang_label = "Chinese" if language == "zh" else "English"
        gender_hint = "男/女" if language == "zh" else "Male/Female"
        prompt = f"""Analyze the following novel excerpt and extract 4-8 main characters.

Output language: {lang_label}
Role code suffix: -{language}

Return a JSON array with these exact keys per element:
[
  {{
    "role_name": "Character name in {lang_label}",
    "role_code": "CharacterEnglishPinyin-{language}",
    "profile": "100-150 char personality/background description in {lang_label}",
    "gender": "{gender_hint}",
    "identity": ["Social role or title in {lang_label}"],
    "relations": {{
      "OtherRoleCode-{language}": {{
        "relation": ["Relationship type in {lang_label}"],
        "detail": "Relationship description in {lang_label}"
      }}
    }}
  }}
]

Rules:
- role_code uses only ASCII letters + hyphen, e.g. JiaBaoyu-zh
- Include relations only for meaningful character pairs you can confirm from the text
- Extract at least 4 characters

Novel excerpt:
{text[:8000]}"""

        return self._parse_json(self._call_llm(prompt))

    def _extract_locations(self, text: str, language: str) -> list:
        lang_label = "Chinese" if language == "zh" else "English"
        prompt = f"""Analyze the following novel excerpt and extract 4-8 main locations.

Output language: {lang_label}

Return a JSON array with these exact keys per element:
[
  {{
    "location_code": "LocationEnglishName",
    "location_name": "Location name in {lang_label}",
    "description": "One-sentence description in {lang_label}",
    "detail": "80-150 char detailed description in {lang_label}"
  }}
]

Rules:
- location_code uses only ASCII letters, no spaces or underscores
- Extract real places mentioned in the novel

Novel excerpt:
{text[:5000]}"""

        return self._parse_json(self._call_llm(prompt))

    # ------------------------------------------------------------ file creation

    def _write_files(self, world_slug: str, world_info: dict,
                     characters: list, locations: list, language: str) -> str:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # -- world_info.json
        world_dir = os.path.join(base, "data", "worlds", world_slug)
        os.makedirs(world_dir, exist_ok=True)
        world_data = {
            "source": world_slug,
            "title": world_info.get("world_name", world_slug),
            "world_name": world_info.get("world_name", world_slug),
            "language": language,
            "description": world_info.get("description", ""),
            "detail": world_info.get("detail", ""),
        }
        with open(os.path.join(world_dir, "world_info.json"), "w", encoding="utf-8") as f:
            json.dump(world_data, f, ensure_ascii=False, indent=2)

        # -- role files
        role_codes = []
        for role in characters:
            code = self._slugify(role.get("role_code", ""))
            if not code:
                continue
            role_dir = os.path.join(base, "data", "roles", world_slug, code)
            os.makedirs(role_dir, exist_ok=True)
            role_data = {
                "role_code": code,
                "role_name": role.get("role_name", code),
                "nickname": role.get("role_name", code),
                "source": world_slug,
                "activity": 1,
                "profile": role.get("profile", ""),
                "gender": role.get("gender", ""),
                "identity": role.get("identity", []),
                "relation": role.get("relations", {}),
            }
            with open(os.path.join(role_dir, "role_info.json"), "w", encoding="utf-8") as f:
                json.dump(role_data, f, ensure_ascii=False, indent=2)
            role_codes.append(code)

        # -- locations.json
        loc_data = {}
        loc_codes = []
        for loc in locations:
            code = self._slugify(loc.get("location_code", ""))
            if not code:
                continue
            loc_data[code] = {
                "location_code": code,
                "location_name": loc.get("location_name", code),
                "source": world_slug,
                "description": loc.get("description", ""),
                "detail": loc.get("detail", ""),
            }
            loc_codes.append(code)
        os.makedirs(os.path.join(base, "data", "locations"), exist_ok=True)
        with open(os.path.join(base, "data", "locations", f"{world_slug}.json"), "w", encoding="utf-8") as f:
            json.dump(loc_data, f, ensure_ascii=False, indent=2)

        # -- map CSV (all distances = 1)
        os.makedirs(os.path.join(base, "data", "maps"), exist_ok=True)
        map_path = os.path.join(base, "data", "maps", f"{world_slug}.csv")
        with open(map_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + loc_codes)
            for src in loc_codes:
                writer.writerow([src] + [0 if src == dst else 1 for dst in loc_codes])

        # -- experiment preset
        preset_name = f"experiment_{world_slug}.json"
        preset_data = {
            "experiment_subname": world_slug,
            "source": world_slug,
            "title": world_info.get("world_name", world_slug),
            "world_file_path": f"./data/worlds/{world_slug}/world_info.json",
            "map_file_path": f"./data/maps/{world_slug}.csv",
            "loc_file_path": f"./data/locations/{world_slug}.json",
            "role_file_dir": "./data/roles/",
            "role_agent_codes": role_codes,
            "intervention": "",
            "script": "",
            "language": language,
        }
        os.makedirs(os.path.join(base, "experiment_presets"), exist_ok=True)
        with open(os.path.join(base, "experiment_presets", preset_name), "w", encoding="utf-8") as f:
            json.dump(preset_data, f, ensure_ascii=False, indent=2)

        return preset_name

    # ------------------------------------------------------------------ public

    def process(self, story_text: str, title: str, language: str = "zh") -> dict:
        """
        Analyse story_text with LLM, write all required data files, and return
        a dict with the new preset filename and extraction summary.
        """
        text = story_text[:10000]

        self._progress("提取世界背景 / Extracting world info", 10)
        world_info = self._extract_world_info(text, title, language)
        world_slug = self._slugify(world_info.get("world_slug", title))

        self._progress("提取角色信息 / Extracting characters", 40)
        characters = self._extract_characters(text, language, f"-{language}")

        self._progress("提取地点信息 / Extracting locations", 70)
        locations = self._extract_locations(text, language)

        self._progress("写入数据文件 / Writing data files", 90)
        preset_name = self._write_files(world_slug, world_info, characters, locations, language)

        self._progress("完成 / Done", 100)
        return {
            "preset": preset_name,
            "world_name": world_info.get("world_name", world_slug),
            "world_slug": world_slug,
            "character_count": len(characters),
            "location_count": len(locations),
        }
