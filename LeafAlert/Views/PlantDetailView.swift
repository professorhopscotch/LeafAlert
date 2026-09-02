import SwiftUI

/// Reference view showing identification details for toxic plants.
struct PlantDetailView: View {
    /// When set, opens straight to that plant's detail (used by the AR sheet's
    /// "Learn about this plant"). Otherwise shows the guide list.
    var selectedPlantID: String? = nil

    // No NavigationStack of its own: this view is always PUSHED — from Home's
    // stack or the AR sheet's. A nested stack inside a navigationDestination
    // makes SwiftUI drop the push, so routing to Plants silently landed on Home.
    var body: some View {
        if let selectedPlantID, let plant = PlantInfo.find(by: selectedPlantID) {
            plantDetail(plant)
        } else {
            List(PlantInfo.all) { plant in
                NavigationLink(value: plant.id) {
                    HStack {
                        Image(systemName: "leaf.fill")
                            .foregroundStyle(plantColor(plant.id))
                        VStack(alignment: .leading) {
                            Text(plant.commonName)
                                .font(.headline)
                            Text(plant.scientificName)
                                .font(.caption)
                                .italic()
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Plant Guide")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: String.self) { plantID in
                if let plant = PlantInfo.find(by: plantID) {
                    plantDetail(plant)
                }
            }
        }
    }

    @ViewBuilder
    private func plantDetail(_ plant: PlantInfo) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text(plant.commonName)
                        .font(.largeTitle.bold())
                    Text(plant.scientificName)
                        .font(.subheadline)
                        .italic()
                        .foregroundStyle(.secondary)
                }

                Divider()

                // Identification by season
                section("Identification by Season") {
                    ForEach(["Spring", "Summer", "Fall", "Winter"], id: \.self) { season in
                        if let tips = plant.identificationTips[season] {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(season)
                                    .font(.subheadline.bold())
                                Text(tips)
                                    .font(.body)
                            }
                        }
                    }
                }

                // Danger
                section("Why It's Dangerous") {
                    Text(plant.dangerDescription)
                }

                // Exposure response
                section("If You're Exposed") {
                    Text(plant.exposureResponse)
                }

                // Look-alikes
                section("Common Look-Alikes") {
                    ForEach(plant.lookAlikes, id: \.self) { lookAlike in
                        Label(lookAlike, systemImage: "arrow.triangle.swap")
                            .font(.body)
                    }
                }

                // Disclaimer
                Text("This app assists identification — always verify visually before touching any plant.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top)
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.title3.bold())
            content()
        }
    }

    private func plantColor(_ id: String) -> Color {
        switch id {
        case "poison_ivy": return .red
        case "poison_oak": return .orange
        case "poison_sumac": return .purple
        default: return .green
        }
    }
}

#Preview {
    NavigationStack {
        PlantDetailView()
    }
}
