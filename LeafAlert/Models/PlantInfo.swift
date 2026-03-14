import Foundation

/// Hardcoded reference information for a toxic plant species.
struct PlantInfo: Identifiable {
    let id: String
    let commonName: String
    let scientificName: String
    let identificationTips: [String: String]  // season → tips
    let dangerDescription: String
    let exposureResponse: String
    let lookAlikes: [String]

    /// All known toxic plants the app can identify.
    static let all: [PlantInfo] = [
        .poisonIvy,
        .poisonOak,
        .poisonSumac
    ]

    static func find(by plantType: String) -> PlantInfo? {
        all.first { $0.id == plantType }
    }

    // MARK: - Plant Definitions

    static let poisonIvy = PlantInfo(
        id: "poison_ivy",
        commonName: "Poison Ivy",
        scientificName: "Toxicodendron radicans",
        identificationTips: [
            "Spring": "Reddish leaves emerging in clusters of three; may have a glossy sheen.",
            "Summer": "Green, almond-shaped leaflets in groups of three; middle leaflet has a longer stem.",
            "Fall": "Leaves turn bright red, orange, or yellow; white berries may be present.",
            "Winter": "Leafless woody vine or shrub with hairy aerial rootlets clinging to trees."
        ],
        dangerDescription: "Contains urushiol oil that causes allergic contact dermatitis in ~85% of people. Rash appears 12–72 hours after exposure and can last 2–3 weeks.",
        exposureResponse: "Immediately wash skin with rubbing alcohol or specialized wash (e.g., Tecnu), then rinse with plenty of water. Do not scratch. Apply calamine lotion or hydrocortisone cream. Seek medical attention for severe reactions or rash near eyes/mouth.",
        lookAlikes: ["Box Elder seedlings", "Virginia Creeper (5 leaflets)", "Fragrant Sumac"]
    )

    static let poisonOak = PlantInfo(
        id: "poison_oak",
        commonName: "Poison Oak",
        scientificName: "Toxicodendron diversilobum",
        identificationTips: [
            "Spring": "New leaves are reddish with a shiny surface; three rounded, lobed leaflets resembling oak leaves.",
            "Summer": "Green leaflets with scalloped edges in groups of three; may form dense shrubs.",
            "Fall": "Leaves turn red, orange, or brown; clusters of small white-green berries.",
            "Winter": "Bare stems with a slightly fuzzy texture; can be hard to identify."
        ],
        dangerDescription: "Contains urushiol oil identical to poison ivy. All parts of the plant are toxic year-round, including stems, roots, and berries.",
        exposureResponse: "Wash exposed area with cold water and soap within 10 minutes if possible. Remove and wash all clothing that contacted the plant. Apply over-the-counter anti-itch treatments. See a doctor if rash is widespread or blistering.",
        lookAlikes: ["True Oak seedlings", "Blackberry bushes", "Desert Mahogany"]
    )

    static let poisonSumac = PlantInfo(
        id: "poison_sumac",
        commonName: "Poison Sumac",
        scientificName: "Toxicodendron vernix",
        identificationTips: [
            "Spring": "Red stems with emerging pinnate leaves; 7–13 smooth-edged leaflets per stem.",
            "Summer": "Tall shrub or small tree with smooth-edged leaflets arranged in pairs along a red stem.",
            "Fall": "Vivid red, orange, and purple foliage; drooping clusters of white berries.",
            "Winter": "Smooth gray bark; may retain white berry clusters. Found almost exclusively in wet, swampy areas."
        ],
        dangerDescription: "Contains more urushiol per leaf than poison ivy or poison oak. Considered the most toxic plant in North America. Thrives in swampy or boggy habitats.",
        exposureResponse: "Rinse skin immediately with lukewarm water and mild soap. Do not use hot water as it can open pores. Wash clothing separately in hot water. Apply cold compresses to affected area. Seek medical help promptly — reactions tend to be more severe than poison ivy.",
        lookAlikes: ["Staghorn Sumac (fuzzy red berries, harmless)", "Tree of Heaven", "Winged Sumac"]
    )
}
