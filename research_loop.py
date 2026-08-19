import os
import re
import time
from datetime import date

import streamlit as st
from groq import Groq
from tavily import TavilyClient


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Dominican Republic B2B Event Research",
    page_icon="🔍",
    layout="wide",
)


# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY.")
    st.stop()

if not tavily_api_key:
    st.error("Missing TAVILY_API_KEY.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)
tavily_client = TavilyClient(api_key=tavily_api_key)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

GENERATOR_MODEL = "openai/gpt-oss-120b"
EVALUATOR_MODEL = "openai/gpt-oss-20b"

MAX_ROUNDS = 4

SEARCH_RESULTS_PER_QUERY = 2
SEARCH_CONTENT_CHARS = 550

TODAY = date.today().isoformat()
CURRENT_YEAR = date.today().year
PREVIOUS_YEAR = CURRENT_YEAR - 1


DEFAULT_QUESTION = """
Evaluate the commercial opportunity to launch a B2B conference and expo in
the Dominican Republic focused on helping businesses get online, sell online,
get found in Google and AI systems, and grow through digital channels.

Determine:

- how the Dominican B2B economy functions today;
- the size and digital maturity of the addressable business market;
- what is changing because of ecommerce, digital payments, search and AI;
- why this matters now;
- whether business owners and executives will pay to attend;
- what comparable events already exist and what market gap remains;
- who the strongest potential sponsors and exhibitors are;
- who credible Dominican speaker candidates are;
- what positioning, city, format, pricing and commercial model are most likely
  to work; and
- what must be validated before significant money is committed.

End with a clear GO, GO BUT VALIDATE FIRST, PILOT ONLY, or NO-GO recommendation.
""".strip()


# --------------------------------------------------
# GROQ HELPER
# --------------------------------------------------

def run_groq(
    model: str,
    prompt: str,
    max_completion_tokens: int,
    reasoning_effort: str = "low",
) -> str:

    for attempt in range(2):

        try:

            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=0,
                reasoning_effort=reasoning_effort,
            )

            return (
                response.choices[0].message.content or ""
            ).strip()

        except Exception as exc:

            message = str(exc).lower()

            # If one individual request is simply too large,
            # waiting will not fix it.
            if "request too large" in message:

                raise RuntimeError(
                    "Groq rejected this request because it is too large "
                    "for the current token-per-minute limit. "
                    "Reduce the search content or completion-token settings."
                ) from exc

            # If we hit a temporary per-minute limit,
            # wait for the rate-limit window and retry once.
            rate_limited = (
                "rate_limit" in message
                or "rate limit" in message
                or "tokens per minute" in message
                or "429" in message
            )

            if rate_limited and attempt == 0:

                time.sleep(65)

                continue

            raise


# --------------------------------------------------
# BASE RESEARCH QUERIES
# --------------------------------------------------

def base_search_queries() -> list[str]:

    return [

        # Size and structure of the Dominican business market
        (
            f"site:one.gob.do empresas MIPYMES "
            f"República Dominicana "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # SME policy and digitalization
        (
            f"site:micm.gob.do MIPYMES digitalización empresas "
            f"República Dominicana "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # Ecommerce and digital payments
        (
            f"site:bancentral.gov.do pagos digitales comercio electrónico "
            f"empresas República Dominicana "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # Existing AI / digital / ecommerce events
        (
            f"República Dominicana congreso evento inteligencia artificial "
            f"marketing digital ecommerce Santo Domingo "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # Ticket prices, sponsors and exhibitors
        (
            f"República Dominicana conferencia empresarial precio entrada "
            f"patrocinadores expositores Santo Domingo "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # Chambers, SMEs and business modernization
        (
            f"República Dominicana cámara de comercio transformación digital "
            f"pymes eventos tecnología "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),

        # International institutional evidence
        (
            f"Dominican Republic SME digital economy ecommerce AI business "
            f"World Bank IDB "
            f"{PREVIOUS_YEAR} {CURRENT_YEAR}"
        ),
    ]


# --------------------------------------------------
# CLEAN SEARCH QUERIES
# --------------------------------------------------

def clean_search_queries(
    queries: list[str]
) -> list[str]:

    cleaned = []

    for query in queries:

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        # Keep Tavily queries concise
        query = query[:380]

        if query and query not in cleaned:
            cleaned.append(query)

    # Gap rounds should stay focused
    return cleaned[:4]


# --------------------------------------------------
# WEB SEARCH
# --------------------------------------------------

def search_web(
    targeted_queries: list[str] | None = None
) -> str:

    if targeted_queries:

        queries = clean_search_queries(
            targeted_queries
        )

    else:

        queries = base_search_queries()

    all_results = []
    seen_urls = set()

    for query in queries:

        try:

            results = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=SEARCH_RESULTS_PER_QUERY,
                include_raw_content=False,
            )

        except Exception as exc:

            all_results.append(
                f"SEARCH QUERY: {query}\n"
                f"SEARCH ERROR: {exc}\n"
            )

            continue

        for result in results.get(
            "results",
            []
        ):

            url = result.get(
                "url",
                ""
            )

            # Do not send the same source repeatedly
            if url and url in seen_urls:
                continue

            if url:
                seen_urls.add(url)

            content = result.get(
                "content",
                ""
            )[:SEARCH_CONTENT_CHARS]

            all_results.append(
                f"SEARCH QUERY: {query}\n"
                f"Source URL: {url}\n"
                f"Title: {result.get('title', '')}\n"
                f"Content: {content}\n"
            )

    return "\n".join(all_results)


# --------------------------------------------------
# GENERATOR
# --------------------------------------------------

def generator(
    question: str,
    search_results: str,
    feedback: str = "",
    previous_answer: str = "",
) -> str:

    revision_context = ""

    if previous_answer:

        revision_context = f"""
PREVIOUS ANSWER

{previous_answer[:7000]}

EDITOR GAPS TO FIX

{feedback}

REVISION INSTRUCTION

Revise the previous answer rather than blindly starting over.

Preserve useful findings that were supported by evidence.

Use the new research to close the gaps identified by the evaluator.

Do not discard good evidence merely because the source did not appear again
in this round of search results.

If new evidence contradicts the previous answer, correct the previous answer
and explain the contradiction where material.
"""

    prompt = f"""
You are a senior market research analyst advising an event investor.

DATE OF RESEARCH:
{TODAY}

THE BUSINESS DECISION

Determine whether there is a commercially viable opportunity for a new B2B
event in the Dominican Republic that helps businesses:

- establish a strong online presence;
- sell online;
- get found in Google;
- get discovered and recommended by AI systems;
- use ecommerce, digital marketing and digital tools effectively; and
- generate business growth through digital channels.

Do NOT try to prove that the event is a good idea.

Your job is to test the opportunity.

If the evidence is weak, say so.

If the event is unlikely to work, recommend NO-GO.


GEOGRAPHY

Focus on the Dominican Republic.

Evaluate Santo Domingo separately as the likely launch market.

Evaluate Santiago where evidence makes it commercially relevant.

Research the market using both Spanish- and English-language evidence.

Do not substitute Latin America-wide statistics for Dominican Republic
statistics unless Dominican-specific information cannot be found.

If regional information is used, label it clearly as regional evidence.


SOURCE STANDARD

Prefer evidence in this order:

1. Dominican government, statistical and regulatory sources
2. Banco Central de la República Dominicana
3. MICM, ONE, INDOTEL and other official institutions
4. Dominican chambers of commerce and recognized business associations
5. World Bank, IDB, ECLAC and other credible international institutions
6. Company reports and official corporate information
7. Original conference and event websites
8. Reputable Dominican journalism
9. Reputable international business journalism

For event pricing, attendance, sponsors, exhibitors and speakers, prioritize
original event and organizer websites.

DO NOT rely on:

- LinkedIn posts
- Medium
- personal blogs
- generic SEO articles
- listicles
- PESTEL or SWOT websites
- unsourced commercial market reports
- promotional claims presented as objective evidence


EVIDENCE RULES

For every important factual claim:

- identify the source;
- identify the date where available;
- use a specific number where reliable data exists;
- include the source URL when it appears in the research material.

Never invent:

- statistics;
- business counts;
- market sizes;
- ticket prices;
- attendance;
- sponsorship prices;
- exhibitor prices;
- sponsor relationships;
- speakers;
- speaker roles; or
- event participation.

Separate conclusions into these evidence standards:

PROVEN
Directly supported by credible evidence.

LIKELY
Supported by evidence but requiring some inference.

UNPROVEN
A reasonable hypothesis that has not been demonstrated.

UNKNOWN
The available research does not answer the question.

Never convert an inference into a fact.

Never claim that AI caused a Dominican market change unless evidence actually
shows that relationship.


1. DOMINICAN B2B ECONOMY

Explain how the Dominican B2B economy functions today.

Investigate where evidence exists:

- number of businesses;
- number and share of SMEs / MIPYMEs;
- formal vs. informal businesses;
- important industries;
- geographic concentration;
- business formation;
- family-owned businesses;
- how companies acquire customers;
- relationship-based selling;
- B2B buying behavior;
- websites;
- WhatsApp;
- social media;
- ecommerce;
- digital payments;
- business software;
- cloud adoption;
- CRM adoption; and
- other relevant business technology.

Do not assume Dominican businesses behave like businesses in the United
States or Canada.


2. DIGITAL MATURITY

Determine how digitally mature Dominican businesses actually are.

Investigate:

- business website adoption;
- ecommerce;
- social commerce;
- digital advertising;
- Google presence;
- online directories;
- digital payments;
- digital banking;
- cloud software;
- CRM;
- AI adoption;
- digital skills gaps;
- SME digital transformation programs.

Look specifically for a gap between how digitally connected Dominican
consumers are and how digitally capable Dominican businesses are.


3. WHAT IS CHANGING

Investigate changes affecting Dominican businesses.

Include where supported:

- ecommerce growth;
- digital payments;
- mobile commerce;
- digital banking;
- search behavior;
- generative AI;
- AI-assisted search;
- business AI adoption;
- AI customer service;
- AI content creation;
- increased digital competition;
- government digitalization programs;
- SME modernization.

Separate:

A. global trends that will affect Dominican businesses;

from:

B. changes already demonstrated inside the Dominican Republic.


4. WHY NOW

Determine whether there is a strong evidence-based reason to launch this
event now.

Look for convergence between:

- business digitalization;
- ecommerce growth;
- digital payments;
- AI adoption;
- changing customer discovery;
- changing Google search;
- SME modernization;
- government programs;
- business competition;
- new technology investment;
- digital skills shortages.

Do not manufacture urgency.

If the "why now" case is weak, say so.


5. ADDRESSABLE EVENT MARKET

Estimate who could realistically attend.

Consider segments such as:

- SME owners;
- family businesses;
- retailers;
- professional services;
- hospitality;
- tourism;
- real estate;
- healthcare;
- exporters;
- manufacturers;
- ecommerce companies;
- entrepreneurs;
- marketing leaders;
- sales leaders;
- commercial leaders.

Do not treat every registered company as an addressable attendee.

Where evidence allows, estimate:

TAM
Total business population relevant to the subject.

SAM
Businesses realistically appropriate for the event.

SOM
A realistic first-event attendee opportunity.

Use ranges rather than false precision.

State every major assumption.


6. WILL THEY PAY?

This question is critical.

Do NOT infer willingness to pay merely from GDP, business count or income.

Look for real Dominican evidence from:

- paid conferences;
- trade shows;
- executive education;
- chamber events;
- workshops;
- marketing events;
- ecommerce conferences;
- AI events;
- technology events;
- entrepreneurship events;
- business seminars.

Determine:

- whether business audiences commonly pay;
- typical ticket prices;
- free vs. paid expectations;
- VIP pricing where available;
- employer-funded attendance;
- what types of business education command payment.

Develop a ticket-pricing hypothesis only after examining comparable evidence.

Clearly label hypotheses.


7. EXISTING EVENTS AND COMPETITION

Identify relevant Dominican events involving:

- ecommerce;
- entrepreneurship;
- SMEs;
- marketing;
- digital transformation;
- technology;
- AI;
- retail;
- tourism technology;
- business growth.

For relevant events identify where available:

- event name;
- organizer;
- city;
- target audience;
- attendance;
- ticket price;
- sponsors;
- exhibitors;
- speakers;
- frequency;
- positioning.

Then determine:

What does the existing event market already cover?

What does it fail to cover?

Would this proposed event occupy a real market gap?

Or would it simply duplicate events that already exist?


8. POTENTIAL SPONSORS

Identify SPECIFIC organizations that could plausibly sponsor the event.

Look for companies whose revenue benefits when Dominican businesses become
more digital.

Potential categories may include:

- banks;
- payment providers;
- telecom companies;
- fintech companies;
- cloud companies;
- software vendors;
- website platforms;
- ecommerce platforms;
- logistics companies;
- accounting software;
- cybersecurity providers;
- digital agencies;
- business associations.

For each serious sponsor candidate provide:

- organization;
- relevant product or business interest;
- evidence of Dominican Republic presence;
- why this audience has commercial value to them;
- evidence of SME programs, business marketing or event sponsorship where
  available;
- recommended sponsorship angle.

Rank them:

TIER 1
Strongest commercial fit.

TIER 2
Credible fit.

TIER 3
Possible, but evidence is weaker.

Do not put a company in Tier 1 simply because it is large or well known.


9. POTENTIAL EXHIBITORS

Keep exhibitors separate from sponsors.

An exhibitor should have something relevant to SELL to the attending
business audience.

For each important exhibitor category explain:

- what they sell;
- why attendees would care;
- what type of buyer they want to meet.

Identify specific companies where supported.

Estimate the plausible exhibitor universe without inventing precise numbers.


10. SPEAKER CANDIDATES

Prioritize speakers based in the Dominican Republic.

Consider:

- successful Dominican business owners;
- ecommerce leaders;
- banking and payment executives;
- telecom executives;
- technology executives;
- government digital-economy leaders;
- chamber and association leaders;
- AI practitioners;
- marketing experts;
- digital transformation leaders.

For every proposed speaker provide:

- name;
- CURRENT role;
- organization;
- why the person fits the event;
- specific topic they could credibly address.

Do not imply that anyone has agreed to speak.

Do not invent availability.

If the current role cannot be verified, do not present the person as a
verified current speaker candidate.


11. EVENT POSITIONING

Do not assume the market will respond to terms such as:

"AI visibility"

"answer engine optimization"

"AI search optimization"

Those may be industry language rather than buyer language.

Test whether the event should instead lead with:

- grow your business online;
- get more customers;
- digital sales;
- ecommerce;
- get found online;
- AI for business;
- digital transformation;
- modernize your business;
- sell in the digital economy.

Recommend the proposition most likely to resonate based on the evidence.


12. EVENT FORMAT

Recommend the strongest format based on the market evidence.

Consider:

- conference;
- expo;
- conference plus expo;
- executive summit;
- SME workshop;
- roadshow;
- chamber partnership;
- hybrid format.

Recommend:

- launch city;
- approximate audience;
- primary attendee;
- Spanish vs. bilingual;
- one-day vs. multi-day;
- conference/expo balance.


13. COMMERCIAL MODEL

Develop a preliminary commercial framework.

Do NOT invent a detailed financial forecast.

Develop evidence-based hypotheses for:

- paid attendance;
- ticket price;
- VIP pricing if appropriate;
- sponsors;
- sponsorship tiers;
- exhibitors;
- exhibitor pricing;
- associations;
- partnerships.

Explain what must be true for the event to be commercially viable.


14. RISKS

Identify credible reasons this event could fail.

Consider:

- businesses expect free education;
- weak paid-event culture;
- sponsors will not pay;
- SME owners are difficult to reach;
- topic is too technical;
- "AI" is the wrong headline;
- existing events already satisfy demand;
- insufficient business density;
- poor event timing;
- digital maturity is too low;
- digital maturity is already too high;
- event does not solve an urgent enough business problem.

Do not minimize negative findings.


15. VALIDATION BEFORE INVESTMENT

Distinguish market research from actual purchase intent.

Recommend what should be tested before committing significant money.

Consider:

- interviews with target attendees;
- sponsor interviews;
- sponsor presales;
- exhibitor presales;
- chamber discussions;
- landing-page demand test;
- email interest campaign;
- ticket-price test;
- refundable early-bird deposits;
- venue hold rather than full commitment.

Specify which tests would materially change the GO / NO-GO decision.


FINAL OUTPUT

Use this exact structure:

### Executive verdict

Choose exactly one:

GO

GO, BUT VALIDATE FIRST

PILOT ONLY

NO-GO

Then explain why.

### Dominican B2B market today

### How businesses currently get customers

### Digital maturity of Dominican businesses

### What is changing

### Why now

### Addressable event market

Include TAM / SAM / SOM where evidence supports them.

### Will businesses pay?

Separate evidence from hypotheses.

### Existing events and the market gap

### Potential sponsors

Include specific organizations and Tier 1 / Tier 2 / Tier 3 rankings.

### Potential exhibitors

Keep exhibitors separate from sponsors.

### Potential speakers

Include names, current roles, organizations and credible topics.

### Recommended positioning

### Recommended event concept

Include city, audience, format, language and duration.

### Preliminary commercial model

### Key risks

### What must be validated before launch

### Top three next actions

### Evidence gaps and confidence

### Key sources

Include the most important source names and URLs available in the research.

Be decisive.

Do not simply summarize the research.

Turn the evidence into a recommendation.

Recommendations must be traceable to evidence rather than generic event,
marketing or digital-transformation best practices.


RESEARCH QUESTION

{question}


NEW RESEARCH MATERIAL

{search_results}


{revision_context}


Write the strongest decision-ready answer supported by the evidence.
"""

    return run_groq(
        model=GENERATOR_MODEL,
        prompt=prompt,
        max_completion_tokens=1600,
        reasoning_effort="low",
    )


# --------------------------------------------------
# EVALUATOR / RED TEAM
# --------------------------------------------------

def evaluator(
    question: str,
    answer: str,
) -> tuple[str, str, list[str]]:

    prompt = f"""
You are a skeptical investment committee reviewing research for a proposed
B2B event in the Dominican Republic.

Your job is NOT to praise the report.

Your job is to determine whether the evidence is strong enough to support a
commercial decision.


RESEARCH QUESTION

{question}


ANSWER TO AUDIT

{answer}


A PASS requires the answer to be materially decision-ready.

Check all of these areas:


1. DOMINICAN-SPECIFIC EVIDENCE

Does the answer rely primarily on evidence from the Dominican Republic rather
than generic Latin American assumptions?


2. B2B MARKET

Does it explain the Dominican business and SME market rather than simply
quoting GDP or population?


3. BUSINESS BEHAVIOR

Does it explain how Dominican businesses currently attract customers, sell,
use digital channels and purchase technology where evidence exists?


4. DIGITAL MATURITY

Does it provide credible evidence about ecommerce, payments, websites,
digital channels, business technology or AI adoption?


5. WHY NOW

Is urgency supported by evidence?

Are global AI trends separated from actual Dominican market evidence?


6. ADDRESSABLE MARKET

Does it develop a defensible TAM / SAM / SOM where possible?

Does it avoid counting every business as a potential attendee?


7. WILLINGNESS TO PAY

Is there actual Dominican evidence from business events, conferences,
training, executive education or comparable paid experiences?

GDP or income alone is NOT evidence of willingness to pay.


8. EVENT COMPETITION

Does the answer identify existing Dominican events that compete for this
audience?


9. MARKET GAP

Does it identify a credible unmet need rather than merely stating that the
market is growing?


10. SPONSORS

Does it identify specific potential sponsors?

Does each important sponsor have a commercial reason to want this audience?

Is Dominican market presence supported?


11. EXHIBITORS

Are exhibitors clearly separated from sponsors?

Does it explain what exhibitors would sell?


12. SPEAKERS

Does it identify credible Dominican speaker candidates?

Are current roles supported?

Does it avoid implying that they have agreed to participate?


13. POSITIONING

Does it test whether "AI" is actually the right headline?

Does it consider business-growth, ecommerce, digital sales and customer
acquisition language?


14. COMMERCIAL MODEL

Does it provide reasonable evidence-backed hypotheses for:

- tickets;
- sponsorship;
- exhibitors;
- format;
- city;
- audience?

Does it avoid fake precision?


15. FAILURE CASE

Does the research seriously consider why the event might fail?


16. VALIDATION

Does it distinguish research from purchase intent?

Does it specify what must be tested before major spending?


17. SOURCE QUALITY

Are important quantitative claims connected to credible named sources?


18. RECENCY

Does the answer use recent evidence wherever available?


19. DECISION

Does it actually choose one:

GO

GO, BUT VALIDATE FIRST

PILOT ONLY

NO-GO


20. EXECUTIVE VALUE

Could an event investor use this information to decide whether to commit
money?


21. MISSING QUESTION

Would an experienced Dominican business owner, sponsor or event organizer
immediately say:

"But you have not considered ______"?


PASS / FAIL RULES

Do not fail because of minor writing or formatting issues.

Fail when important evidence, analysis or commercial reasoning is missing.

If the answer is genuinely decision-ready, respond exactly:

VERDICT: PASS


If important gaps remain, respond:

VERDICT: FAIL

GAPS:
- [specific missing evidence or weakness]
- [specific missing evidence or weakness]
- [specific missing evidence or weakness]

SEARCH_QUERIES:
- [one exact search query aimed at the highest-value missing evidence]
- [second exact search query]
- [third exact search query if useful]
- [fourth exact search query if useful]


SEARCH QUERY RULES

Make the searches:

- concise;
- specific;
- focused on the Dominican Republic;
- focused on the evidence actually missing.

Use Spanish when Spanish is more likely to surface local evidence.

Do not give vague feedback such as:

"needs more detail"

Tell the researcher exactly what evidence to investigate next.
"""

    evaluation = run_groq(
        model=EVALUATOR_MODEL,
        prompt=prompt,
        max_completion_tokens=650,
        reasoning_effort="low",
    )

    if "VERDICT: PASS" in evaluation:

        return "PASS", "", []

    gaps_match = re.search(
        r"GAPS:(.*?)(?:SEARCH_QUERIES:|$)",
        evaluation,
        re.DOTALL,
    )

    if gaps_match:

        gaps = gaps_match.group(1).strip()

    else:

        gaps = (
            "The evaluator failed the answer but did not specify the gaps."
        )

    queries_match = re.search(
        r"SEARCH_QUERIES:(.*)",
        evaluation,
        re.DOTALL,
    )

    search_queries = []

    if queries_match:

        for line in queries_match.group(1).splitlines():

            line = re.sub(
                r"^\s*[-*]\s*",
                "",
                line,
            ).strip()

            if line:
                search_queries.append(line)

    return (
        "FAIL",
        gaps,
        clean_search_queries(search_queries),
    )


# --------------------------------------------------
# RESEARCH LOOP
# --------------------------------------------------

def research_loop(
    question: str,
    status_container,
    output_container,
):

    previous_answer = ""
    feedback = ""
    targeted_queries = None

    final_answer = ""
    remaining_gaps = ""

    for round_num in range(
        1,
        MAX_ROUNDS + 1,
    ):

        status_container.markdown(
            f"**Round {round_num}** - "
            "Researching the Dominican market..."
        )

        search_results = search_web(
            targeted_queries
        )

        status_container.markdown(
            f"**Round {round_num}** - "
            "Building the market assessment..."
        )

        answer = generator(
            question=question,
            search_results=search_results,
            feedback=feedback,
            previous_answer=previous_answer,
        )

        status_container.markdown(
            f"**Round {round_num}** - "
            "Red-teaming the evidence and recommendations..."
        )

        verdict, gaps, next_queries = evaluator(
            question=question,
            answer=answer,
        )

        final_answer = answer

        if verdict == "PASS":

            status_container.markdown(
                f"**Round {round_num}** - ✅ Decision-ready"
            )

            remaining_gaps = ""

            break

        remaining_gaps = gaps

        status_container.markdown(
            f"**Round {round_num}** - ❌ Evidence gaps found:\n\n"
            f"{gaps}"
        )

        previous_answer = answer
        feedback = gaps

        if next_queries:

            targeted_queries = next_queries

        else:

            targeted_queries = None

        if round_num == MAX_ROUNDS:

            status_container.markdown(
                f"**Round {round_num}** - ⚠️ Max rounds reached. "
                "Returning the strongest answer with unresolved "
                "evidence gaps clearly flagged."
            )

    output_container.markdown(
        "## Research Complete"
    )

    output_container.markdown(
        final_answer
    )

    if remaining_gaps:

        output_container.markdown(
            "## Remaining Validation Gaps"
        )

        output_container.markdown(
            remaining_gaps
        )


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.title(
    "🔍 Dominican Republic B2B Event Research Loop"
)

st.markdown(
    "Tests the commercial opportunity for a Dominican Republic "
    "B2B event focused on helping businesses get online, get "
    "found in search and AI, sell digitally and grow."
)

question = st.text_area(
    "Research brief",
    value=DEFAULT_QUESTION,
    height=340,
)

if st.button(
    "Run Research Loop",
    type="primary",
):

    if not question.strip():

        st.error(
            "Please enter a research question."
        )

    else:

        st.markdown("---")

        st.markdown(
            "**Loop Progress**"
        )

        status_container = st.empty()

        st.markdown("---")

        output_container = st.container()

        with st.spinner(
            "Running Dominican Republic market research..."
        ):

            research_loop(
                question,
                status_container,
                output_container,
            )
