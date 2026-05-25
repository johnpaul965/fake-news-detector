export interface CorroboratingSource {
  title: string;
  url: string;
  snippet: string;
  publisher: string;
  type: "credible" | "factcheck";
}

export function buildSearchLinks(
  title: string,
  prediction: "fake" | "real"
): CorroboratingSource[] {
  const keywords = title
    .split(" ")
    .slice(0, 7)
    .join(" ")
    .replace(/[^a-zA-Z0-9 ]/g, "")
    .trim();

  const q = encodeURIComponent(keywords);

  if (prediction === "real") {
    return [
      {
        title: `Search "${keywords}" on Inquirer`,
        url: `https://newsinfo.inquirer.net/?s=${q}`,
        snippet: "Philippine Daily Inquirer — one of the Philippines' most trusted broadsheets",
        publisher: "inquirer.net",
        type: "credible",
      },
      {
        title: `Search "${keywords}" on Rappler`,
        url: `https://www.rappler.com/search/#stq=${q}`,
        snippet: "Rappler — award-winning Philippine digital news outlet",
        publisher: "rappler.com",
        type: "credible",
      },
      {
        title: `Search "${keywords}" on GMA News`,
        url: `https://www.gmanetwork.com/news/search/?q=${q}`,
        snippet: "GMA Network News — major Philippine broadcast network",
        publisher: "gmanetwork.com",
        type: "credible",
      },
      {
        title: `Search "${keywords}" on Philippine News Agency`,
        url: `https://www.pna.gov.ph/?s=${q}`,
        snippet: "PNA — official Philippine government news agency",
        publisher: "pna.gov.ph",
        type: "credible",
      },
      {
        title: `Search "${keywords}" on Manila Bulletin`,
        url: `https://mb.com.ph/?s=${q}`,
        snippet: "Manila Bulletin — one of the Philippines' oldest newspapers",
        publisher: "mb.com.ph",
        type: "credible",
      },
    ];
  } else {
    return [
      {
        title: `Search "${keywords}" on VERA Files`,
        url: `https://verafiles.org/?s=${q}`,
        snippet: "VERA Files — Philippine fact-checking organization accredited by IFCN",
        publisher: "verafiles.org",
        type: "factcheck",
      },
      {
        title: `Search "${keywords}" on AFP Fact Check Philippines`,
        url: `https://factcheck.afp.com/?s=${q}`,
        snippet: "AFP Fact Check — global newswire fact-checking team covering the Philippines",
        publisher: "factcheck.afp.com",
        type: "factcheck",
      },
      {
        title: `Search "${keywords}" on Rappler Fact Check`,
        url: `https://www.rappler.com/fact-check/?s=${q}`,
        snippet: "Rappler Fact Check — Philippine digital news outlet's dedicated fact-checking section",
        publisher: "rappler.com",
        type: "factcheck",
      },
      {
        title: `Search "${keywords}" on TSEK.PH`,
        url: `https://tsek.ph/?s=${q}`,
        snippet: "TSEK.PH — collaborative Philippine fact-checking platform",
        publisher: "tsek.ph",
        type: "factcheck",
      },
      {
        title: `Search "${keywords}" on PolitiFact`,
        url: `https://www.politifact.com/search/?q=${q}`,
        snippet: "PolitiFact — Pulitzer Prize-winning international fact-checking organization",
        publisher: "politifact.com",
        type: "factcheck",
      },
    ];
  }
}

export async function findCorroboratingSources(
  title: string,
  prediction: "fake" | "real"
): Promise<CorroboratingSource[]> {
  return buildSearchLinks(title, prediction);
}
