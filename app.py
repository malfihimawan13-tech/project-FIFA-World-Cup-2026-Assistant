import os
import re
import streamlit as st
from datetime import date

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# ==================================
# LOAD ENV
# ==================================

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="World Cup 2026 Assistant",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA World Cup 2026 Assistant")
st.caption("Llama 3 + FAISS + LangChain + RAG")

# ==================================
# SESSION STATE
# ==================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "squad_retriever" not in st.session_state:
    st.session_state.squad_retriever = None

if "schedule_text" not in st.session_state:
    st.session_state.schedule_text = None

# ==================================
# HELPERS
# ==================================

TEAM_HEADER_RE = re.compile(r"^[A-Za-zÀ-ÿ' .]+ \([A-Z]{3}\)$")


def hitung_umur(dob_str: str) -> int:
    try:
        day, month, year = map(int, dob_str.split("/"))
        born = date(year, month, day)
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


def bersihkan_nama_coach(line: str) -> str:
    line = re.sub(r"(?i)head\s*coach\s*", "", line).strip()
    line = re.sub(r"\s+", " ", line)
    parts = line.split()
    if len(parts) < 3:
        return line

    # last_name = kata pertama (ditulis ALL CAPS)
    last_name = parts[0]

    # nationality = kata/frasa terakhir (bisa lebih dari 1 kata, mis. "Bosnia And Herzegovina")
    # Kita asumsikan first_name terdiri dari kata2 setelah parts[1] (first name pendek)
    # sampai sebelum kemunculan ulang last_name (karena formatnya: LASTNAME firstname Fullname LASTNAME Nationality)
    # Cara aman: cari index kemunculan kedua last_name (case-insensitive, abaikan diakritik kasar)
    last_upper = last_name.upper()
    second_idx = None
    for i in range(1, len(parts)):
        if parts[i].upper().replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U").startswith(last_upper[:4]):
            second_idx = i
            break

    if second_idx and second_idx + 1 <= len(parts):
        first_name = " ".join(parts[1:second_idx])
        nationality = " ".join(parts[second_idx+1:])
    else:
        first_name = parts[1]
        nationality = " ".join(parts[2:])

    if first_name and last_name:
        if nationality:
            return f"{first_name} {last_name} ({nationality})"
        return f"{first_name} {last_name}"
    return line


# ==================================
# LOADER SKUAD (SAMA DENGAN test_parse.py YANG SUDAH TERBUKTI BENAR)
# ==================================

def normalize_header(cell: str) -> str:
    return re.sub(r"\s+", " ", (cell or "").strip().upper())

PLAYER_HEADER_KEYS = {
    "POS": "POS",
    "FIRST NAME(S)": "FIRST",
    "LAST NAME(S)": "LAST",
    "DOB": "DOB",
    "CLUB": "CLUB",
    "CAPS": "CAPS",
    "GOALS": "GOALS",
}

def load_squad_documents(path: str, debug: bool = False):
    import pdfplumber
    docs = []
    debug_info = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            team_name = None
            for line in text.split("\n"):
                line = line.strip()
                if TEAM_HEADER_RE.match(line):
                    team_name = line
                    break
            if not team_name:
                continue

            tables = page.extract_tables()
            if not tables:
                continue

            n_matched = 0
            for table in tables:
                col_map = {}
                for row in table:
                    if not row:
                        continue
                    row = [str(c).strip() if c else "" for c in row]
                    norm_cells = [normalize_header(c) for c in row]

                    # Deteksi baris header tabel pemain
                    if "POS" in norm_cells and "DOB" in norm_cells:
                        col_map = {}
                        for key, short in PLAYER_HEADER_KEYS.items():
                            if key in norm_cells:
                                col_map[short] = norm_cells.index(key)
                        continue

                    if not col_map or len(col_map) < len(PLAYER_HEADER_KEYS):
                        continue

                    max_idx = max(col_map.values())
                    if len(row) <= max_idx:
                        continue

                    pos        = row[col_map["POS"]]
                    first_name = row[col_map["FIRST"]]
                    last_name  = row[col_map["LAST"]]
                    dob_str    = row[col_map["DOB"]]
                    club       = row[col_map["CLUB"]]
                    caps       = row[col_map["CAPS"]]
                    goals      = row[col_map["GOALS"]]

                    if pos not in ("GK", "DF", "MF", "FW"):
                        continue
                    if not re.match(r"\d{2}/\d{2}/\d{4}", dob_str):
                        continue

                    caps  = caps  if caps.isdigit()  else "0"
                    goals = goals if goals.isdigit() else "0"

                    nama = f"{first_name} {last_name}".strip()
                    umur = hitung_umur(dob_str)

                    content = (
                        f"Tim: {team_name}. "
                        f"Posisi: {pos}. "
                        f"Pemain: {nama}. "
                        f"Umur: {umur} tahun. "
                        f"Klub: {club}. "
                        f"Caps: {caps}. Gol: {goals}."
                    )
                    docs.append(Document(
                        page_content=content,
                        metadata={"team": team_name, "source": path}
                    ))
                    n_matched += 1

            if debug:
                debug_info.append(f"[{team_name}] page={page_num+1} matched={n_matched}")

            for table in tables:
                coach_col_map = {}
                for row in table:
                    if not row:
                        continue
                    row = [str(c).strip() if c else "" for c in row]
                    norm_cells = [normalize_header(c) for c in row]

                    if "COACH NAME" in norm_cells and "NATIONALITY" in norm_cells:
                        coach_col_map = {
                            "FIRST": norm_cells.index("FIRST NAME(S)"),
                            "LAST": norm_cells.index("LAST NAME(S)"),
                            "NAT": norm_cells.index("NATIONALITY"),
                        }
                        continue

                    if coach_col_map and len(row) > max(coach_col_map.values()):
                        first = row[coach_col_map["FIRST"]]
                        last  = row[coach_col_map["LAST"]]
                        nat   = row[coach_col_map["NAT"]]
                        if first and last:
                            nama_coach = f"{first} {last} ({nat})"
                            docs.append(Document(
                                page_content=(
                                    f"Tim: {team_name}. "
                                    f"Pelatih kepala: {nama_coach}. "
                                    f"Pelatih tim {team_name} adalah {nama_coach}."
                                ),
                                metadata={"team": team_name, "source": path}
                            ))
                            coach_col_map = {}

    if debug:
        return docs, debug_info
    return docs


# ==================================
# LOAD SCHEDULE
# ==================================

def load_schedule_full_text(path: str) -> str:
    pages = PyPDFLoader(path).load()
    return "\n".join(p.page_content for p in pages)


# ==================================
# INITIALIZE (selalu bangun ulang dari PDF, tidak load index lama)
# ==================================

if st.session_state.squad_retriever is None or st.session_state.schedule_text is None:

    with st.spinner("Memuat data World Cup 2026..."):

        squad_docs = load_squad_documents("data/SquadLists-English.pdf")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        squad_vs = FAISS.from_documents(squad_docs, embeddings)

        st.session_state.squad_vectorstore = squad_vs
        st.session_state.squad_docs = squad_docs

        st.session_state.squad_retriever = squad_vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )

        st.session_state.schedule_text = load_schedule_full_text(
            "data/FWC26 Match Schedule_v17_10042026_EN.pdf"
        )

# ==================================
# SIDEBAR RESET + DEBUG TOOL
# ==================================

with st.sidebar:
    st.header("⚙️ Pengaturan")
    if st.button("🔄 Reset & Reload Data"):
        st.session_state.squad_retriever = None
        st.session_state.squad_vectorstore = None
        st.session_state.squad_docs = None
        st.session_state.schedule_text = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("🔍 Debug Parsing Skuad")

    total_docs = len(st.session_state.get("squad_docs", []))
    arg_count = sum(
        1 for d in st.session_state.get("squad_docs", [])
        if "Argentina" in d.metadata.get("team", "") and "Pelatih" not in d.page_content
    )
    st.caption(f"Total dokumen di index: {total_docs}")
    st.caption(f"Dokumen pemain Argentina: {arg_count}")

    if st.button("Cek hasil parsing per tim"):
        docs_debug, debug_info = load_squad_documents(
            "data/SquadLists-English.pdf", debug=True
        )
        from collections import Counter
        team_counts = Counter()
        for d in docs_debug:
            if "Pelatih" not in d.page_content:
                team_counts[d.metadata["team"]] += 1

        st.write("**Jumlah pemain ter-parse per tim:**")
        for team, count in sorted(team_counts.items()):
            flag = "✅" if count == 26 else "⚠️"
            st.write(f"{flag} {team}: {count} pemain")

        st.write("**Detail token gagal / ringkasan:**")
        for line in debug_info:
            st.text(line)

# ==================================
# SHOW HISTORY
# ==================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================================
# ROUTING
# ==================================

SCHEDULE_KEYWORDS = [
    "lawan", "vs", "pertandingan", "jadwal", "kapan",
    "venue", "stadion", "grup", "group", "tanggal", "main",
    "tanding", "babak", "jam", "melawan", "bermain",
]

TEAM_MAP = {
    "argentina": "ARG", "brazil": "BRA", "brasil": "BRA",
    "spanyol": "ESP", "spain": "ESP", "portugal": "POR",
    "prancis": "FRA", "france": "FRA", "perancis": "FRA", "jerman": "GER",
    "germany": "GER", "inggris": "ENG", "england": "ENG",
    "italia": "ITA", "italy": "ITA", "belanda": "NED",
    "netherlands": "NED", "belgia": "BEL", "belgium": "BEL",
    "uruguay": "URU", "meksiko": "MEX", "mexico": "MEX",
    "kroasia": "CRO", "croatia": "CRO", "senegal": "SEN",
    "maroko": "MAR", "morocco": "MAR", "jepang": "JPN",
    "japan": "JPN", "korea": "KOR", "australia": "AUS",
    "kanada": "CAN", "canada": "CAN", "swiss": "SUI", "switzerland": "SUI",
    "austria": "AUT", "norwegia": "NOR", "norway": "NOR",
    "skotlandia": "SCO", "scotland": "SCO", "irak": "IRQ",
    "iraq": "IRQ", "algeria": "ALG", "aljazair": "ALG",
    "kolombia": "COL", "colombia": "COL", "ekuador": "ECU",
    "ecuador": "ECU", "turki": "TUR", "turkey": "TUR", "türkiye": "TUR",
    "swedia": "SWE", "sweden": "SWE", "denmark": "DEN",
    "polandia": "POL", "poland": "POL", "ghana": "GHA",
    "nigeria": "NGA", "kamerun": "CMR", "cameroon": "CMR",
    "mesir": "EGY", "egypt": "EGY", "tunisia": "TUN",
    "arab saudi": "KSA", "saudi": "KSA", "saudi arabia": "KSA",
    "iran": "IRN", "china": "CHN",
    "amerika serikat": "USA", "amerika": "USA", "usa": "USA",
    "kosta rika": "CRC", "panama": "PAN", "haiti": "HAI",
    "paraguay": "PAR", "chile": "CHI",
    "peru": "PER", "venezuela": "VEN", "bolivia": "BOL",
    "selandia baru": "NZL", "new zealand": "NZL",
    "bosnia": "BIH", "bosnia dan herzegovina": "BIH",
    "kongo": "COD", "congo": "COD", "republik kongo": "COD",
    "uzbekistan": "UZB", "jordania": "JOR", "yordania": "JOR", "jordan": "JOR",
    "cabo verde": "CPV", "curacao": "CUW", "curaçao": "CUW",
    "czechia": "CZE", "ceko": "CZE", "czech republic": "CZE",
    "qatar": "QAT", "south africa": "RSA", "afrika selatan": "RSA",
    "côte d'ivoire": "CIV", "ivory coast": "CIV", "pantai gading": "CIV",
}


def extract_team_code(question: str) -> str | None:
    q_lower = question.lower()
    # Urutkan keyword terpanjang dulu agar "amerika serikat" tidak
    # ke-match jadi "amerika" saja, dst.
    for keyword in sorted(TEAM_MAP.keys(), key=len, reverse=True):
        if keyword in q_lower:
            return TEAM_MAP[keyword]

    import difflib
    import string
    words = q_lower.split()
    for word in words:
        word_clean = word.strip(string.punctuation)
        if len(word_clean) < 4:
            continue
        matches = difflib.get_close_matches(word_clean, TEAM_MAP.keys(), n=1, cutoff=0.75)
        if matches:
            return TEAM_MAP[matches[0]]
    return None


def detect_team_filter(question: str, retriever):
    q_lower = question.lower()
    team_code = extract_team_code(question)

    if team_code:
        # Ambil SEMUA dokumen dari vectorstore, filter HANYA tim yang dimaksud.
        # Gunakan regex agar "ARG" tidak ke-match ke kode tim lain yang
        # mengandung substring serupa (mis. "ARG" vs negara lain).
        all_docs = list(retriever.vectorstore.docstore._dict.values())

        team_docs = [
            doc for doc in all_docs
            if doc.metadata.get("team", "").endswith(f"({team_code})")
        ]

        pos_filter = None
        if any(k in q_lower for k in ["kiper", "goalkeeper", "gk", "penjaga gawang"]):
            pos_filter = "Posisi: GK"
        elif any(k in q_lower for k in ["bek", "defender", "df", "pertahanan"]):
            pos_filter = "Posisi: DF"
        elif any(k in q_lower for k in ["gelandang", "midfielder", "mf"]):
            pos_filter = "Posisi: MF"
        elif any(k in q_lower for k in ["penyerang", "striker", "fw", "forward"]):
            pos_filter = "Posisi: FW"

        if pos_filter and team_docs:
            pos_docs = [d for d in team_docs if pos_filter in d.page_content]
            if pos_docs:
                return pos_docs

        if team_docs:
            coach_docs = [d for d in team_docs if "Pelatih" in d.page_content]
            player_docs = [d for d in team_docs if "Pelatih" not in d.page_content]
            return coach_docs + player_docs

    # Fallback similarity search (hanya kalau tim tidak terdeteksi)
    if any(k in q_lower for k in ["kiper", "goalkeeper", "gk", "penjaga gawang"]):
        results = retriever.vectorstore.similarity_search_with_score(
            question + " GK goalkeeper posisi", k=20
        )
    else:
        results = retriever.vectorstore.similarity_search_with_score(question, k=15)

    threshold = 1.5
    docs = [doc for doc, score in results if score < threshold]

    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs if unique_docs else [doc for doc, _ in results[:5]]


def build_context(question: str) -> str:
    q_lower = question.lower()
    use_schedule = any(k in q_lower for k in SCHEDULE_KEYWORDS)

    parts = []

    if use_schedule:
        parts.append("=== JADWAL PERTANDINGAN ===\n" + st.session_state.schedule_text[:8000])

    squad_docs = detect_team_filter(question, st.session_state.squad_retriever)
    if squad_docs:
        squad_text = "\n".join(d.page_content for d in squad_docs)
        parts.append("=== DATA PEMAIN ===\n" + squad_text[:30000])

    return "\n\n".join(parts)


def is_context_useful(context: str) -> bool:
    if not context.strip():
        return False
    words = context.split()
    if len(words) < 5:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.1:
        return False
    return True

# ==================================
# USER INPUT
# ==================================

question = st.chat_input("Tanyakan sesuatu tentang World Cup 2026...")

def format_squad_list(squad_docs, team_label):
    """Format daftar skuad langsung dari dokumen, tanpa LLM."""
    player_docs = [d for d in squad_docs if "Pelatih" not in d.page_content]
    coach_docs = [d for d in squad_docs if "Pelatih" in d.page_content]

    lines = []
    if coach_docs:
        m = re.search(r"Pelatih kepala: (.+?)\.", coach_docs[0].page_content)
        if m:
            lines.append(f"**Pelatih {team_label}:** {m.group(1)}")
            lines.append("")

    lines.append(f"**Skuad {team_label}:**")
    lines.append("")
    for i, d in enumerate(player_docs, 1):
        m = re.search(
            r"Posisi: (\w+)\. Pemain: (.+?)\. Umur: (\d+) tahun\. Klub: (.+?)\.",
            d.page_content
        )
        if m:
            pos, nama, umur, klub = m.groups()
            lines.append(f"{i}. {pos} | {nama} | {umur} th | {klub}")

    return "\n".join(lines)

# ==================================
# PLAYER STATUS CHECK
# ==================================

import unicodedata

def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()

PLAYER_CHECK_PATTERNS = [
    "masih masuk", "masih main", "masih bermain", "masih termasuk",
    "masih terdaftar", "masuk skuad", "termasuk skuad",
    "terdaftar di skuad", "ada di skuad", "main untuk",
    "bermain untuk", "masih dipanggil", "masih ada",
    "masih di", "masih main di", "masih bermain di",
    "masih membela", "masih terpilih",
]

def is_player_status_question(q_lower: str) -> bool:
    return any(p in q_lower for p in PLAYER_CHECK_PATTERNS)

def find_player_in_team(question: str, team_docs):
    q_norm = normalize_name(question)

    player_docs = [
        d for d in team_docs
        if "Pelatih" not in d.page_content
    ]

    for d in player_docs:

        m = re.search(
            r"Posisi: (\w+)\. Pemain: (.+?)\. Umur: (\d+) tahun\. Klub: (.+?)\. Caps: (\d+)\. Gol: (\d+)\.",
            d.page_content
        )

        if not m:
            continue

        pos, nama, umur, klub, caps, gol = m.groups()

        for token in nama.replace(".", "").split():

            if (
                len(token) >= 3
                and normalize_name(token) in q_norm
            ):
                return {
                    "found": True,
                    "nama": nama,
                    "pos": pos,
                    "umur": umur,
                    "klub": klub,
                    "caps": caps,
                    "gol": gol
                }

    return {"found": False}

SQUAD_LIST_KEYWORDS = [
    "skuad", "squad", "daftar pemain", "siapa saja pemain",
    "pemain skuad", "list pemain", "semua pemain"
]

NON_LIST_QUESTION_WORDS = [
    "apakah", "masih", "berapa", "kapan", "umur", "usia",
    "caps", "gol", "tinggi", "klub", "main untuk", "bermain untuk"
]


if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    q_lower = question.lower()
    team_code = extract_team_code(question)

    is_squad_list_request = (
        any(k in q_lower for k in SQUAD_LIST_KEYWORDS)
        and not any(w in q_lower for w in NON_LIST_QUESTION_WORDS)
    )

    if team_code and is_squad_list_request:
        # Jawab langsung dari data, tanpa LLM, biar tidak halusinasi
        all_docs = list(st.session_state.squad_retriever.vectorstore.docstore._dict.values())
        team_docs = [
            doc for doc in all_docs
            if doc.metadata.get("team", "").endswith(f"({team_code})")
        ]
        team_label = team_docs[0].metadata.get("team", team_code) if team_docs else team_code
        answer = format_squad_list(team_docs, team_label)

    elif team_code and is_player_status_question(q_lower):

        all_docs = list(
            st.session_state.squad_retriever
            .vectorstore.docstore._dict.values()
        )

        team_docs = [
            doc for doc in all_docs
            if doc.metadata.get("team", "")
            .endswith(f"({team_code})")
        ]

        team_label = (
            team_docs[0].metadata.get("team", team_code)
            if team_docs else team_code
        )

        result = find_player_in_team(
            question,
            team_docs
        )

        if result["found"]:

            answer = (
                f"Ya, {result['nama']} "
                f"({result['pos']}) masuk skuad "
                f"{team_label} untuk Piala Dunia 2026. "
                f"Saat ini bermain untuk {result['klub']} "
                f"dan berusia {result['umur']} tahun."
            )

        else:

            answer = (
                f"Pemain tersebut tidak ditemukan "
                f"di skuad {team_label}."
            )
    
    
    else:
        context = build_context(question)

        if not is_context_useful(context):
            answer = (
                "Maaf, saya tidak menemukan informasi tersebut di data yang saya miliki. "
                "Coba sebutkan nama tim, pemain, atau tanggal secara spesifik."
            )

        else:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0,
                max_tokens=4000,
            )

            prompt = f"""
Anda adalah asisten FIFA World Cup 2026. Jawab dalam Bahasa Indonesia, singkat dan jelas.

ATURAN MENJAWAB (SANGAT PENTING):
- KONTEKS di bawah adalah SATU-SATUNYA sumber data yang valid dan terkini (skuad Piala Dunia 2026).
- Pengetahuan internal Anda tentang skuad timnas (termasuk skuad 2022, 2018, atau tahun lain) SUDAH USANG dan TIDAK BOLEH digunakan sama sekali.
- DILARANG menambahkan, mengganti, atau menyebut nama pemain MANAPUN yang TIDAK tertulis secara eksplisit di KONTEKS.
- UNTUK PERTANYAAN "apakah [nama pemain] masih/termasuk/masuk skuad [tim]?":
  Cari baris "Pemain: [nama]" di KONTEKS bagian DATA PEMAIN untuk tim tersebut.
  - Jika DITEMUKAN, jawab: "Ya, [Nama Lengkap] ([Posisi]) masuk skuad [Tim] untuk Piala Dunia 2026, bermain di klub [Klub]."
  - Jika TIDAK DITEMUKAN di daftar pemain tim tersebut, jawab: "Tidak, [nama] tidak terdaftar di skuad [Tim] untuk Piala Dunia 2026 berdasarkan data yang saya miliki."
- Untuk pertanyaan pelatih, jawab ringkas satu kalimat:
  "Pelatih [tim] adalah [Nama Depan NAMA BELAKANG] ([negara asal])."
- Untuk pertanyaan satu pemain (caps, gol, umur, klub), jawab ringkas satu kalimat.
- Jika dua hal spesifik tidak berkaitan di konteks, jawab TIDAK ADA dengan penjelasan singkat.
- Jika konteks tidak relevan sama sekali, jawab "Maaf, saya tidak menemukan informasi tersebut."

CARA MEMBACA JADWAL:
- Format: "TEAMv TEAM Grup No Jam(ET)"
  Contoh: "FRAv IRQ I 42 17:00" = France vs Iraq, Grup I, match #42, 17:00 ET.
- Jika dua tim tidak saling berhadapan di jadwal, jawab TIDAK ADA dan sebutkan grup masing-masing.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""

            response = llm.invoke(prompt)
            answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})