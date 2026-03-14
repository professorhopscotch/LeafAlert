import SwiftUI

/// Reference view showing identification details for toxic plants.
struct PlantDetailView: View {
    var selectedPlantID: String? = nil

    @State private var selectedPlant: PlantInfo?

    var body: some View {
        List(PlantInfo.all) { plant in
            NavigationLink(destination: plantDetail(plant)) {
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
        .onAppear {
            if let selectedPlantID {
                selectedPlant = PlantInfo.find(by: selectedPlantID)
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
