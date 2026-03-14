import SwiftUI
import MapKit
import SwiftData

/// Displays all logged detections as color-coded pins on a MapKit map.
/// Works fully offline — no network tiles needed for base map.
struct PatrolMapView: View {
    @Query(sort: \DetectionLog.timestamp, order: .reverse) private var logs: [DetectionLog]
    @Environment(\.modelContext) private var modelContext

    @State private var selectedLog: DetectionLog?
    @State private var filterMode: FilterMode = .all

    enum FilterMode: String, CaseIterable {
        case all = "Show All"
        case confirmed = "Confirmed Only"
        case unconfirmed = "Unconfirmed Only"
    }

    private var filteredLogs: [DetectionLog] {
        switch filterMode {
        case .all:
            return logs
        case .confirmed:
            return logs.filter { $0.userConfirmed == true }
        case .unconfirmed:
            return logs.filter { $0.userConfirmed != true }
        }
    }

    var body: some View {
        ZStack(alignment: .top) {
            Map {
                ForEach(filteredLogs) { log in
                    Annotation(
                        DetectionFormatting.plantDisplayName(log.plantType),
                        coordinate: CLLocationCoordinate2D(
                            latitude: log.latitude,
                            longitude: log.longitude
                        )
                    ) {
                        pinView(for: log)
                            .onTapGesture {
                                selectedLog = log
                            }
                    }
                }
            }

            filterPicker
        }
        .navigationTitle("Detection Map")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if logs.isEmpty {
                ContentUnavailableView(
                    "No Detections Yet",
                    systemImage: "map",
                    description: Text("Start a patrol to begin logging detections.")
                )
            }
        }
        .sheet(item: $selectedLog) { log in
            detailPopup(for: log)
        }
    }

    // MARK: - Filter Picker

    private var filterPicker: some View {
        Picker("Filter", selection: $filterMode) {
            ForEach(FilterMode.allCases, id: \.self) { mode in
                Text(mode.rawValue).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .padding(.top, 8)
    }

    // MARK: - Detail Popup

    private func detailPopup(for log: DetectionLog) -> some View {
        VStack(spacing: 16) {
            Text(DetectionFormatting.plantDisplayName(log.plantType))
                .font(.title2.bold())

            HStack(spacing: 24) {
                Label(
                    "\(Int(log.confidence * 100))%",
                    systemImage: "gauge.medium"
                )
                .foregroundStyle(DetectionFormatting.confidenceColor(log.confidence))

                Label(
                    DetectionFormatting.relativeTimestamp(log.timestamp),
                    systemImage: "clock"
                )
                .foregroundStyle(.secondary)
            }
            .font(.subheadline)

            if let confirmed = log.userConfirmed {
                Label(
                    confirmed ? "Confirmed" : "Dismissed",
                    systemImage: confirmed ? "checkmark.circle.fill" : "xmark.circle.fill"
                )
                .foregroundStyle(confirmed ? .green : .red)
                .font(.subheadline)
            }

            Divider()

            HStack(spacing: 16) {
                Button {
                    updateConfirmation(log: log, confirmed: true)
                    selectedLog = nil
                } label: {
                    Label("Confirm", systemImage: "checkmark.circle")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)

                Button {
                    updateConfirmation(log: log, confirmed: false)
                    selectedLog = nil
                } label: {
                    Label("Dismiss", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.bordered)
            }

            Text(DetectionFormatting.safetyDisclaimer)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .presentationDetents([.medium])
    }

    // MARK: - Pin View

    @ViewBuilder
    private func pinView(for log: DetectionLog) -> some View {
        Circle()
            .fill(DetectionFormatting.confidenceColor(log.confidence))
            .frame(width: 14, height: 14)
            .overlay(
                Circle().stroke(.white, lineWidth: 2)
            )
    }

    // MARK: - Helpers

    private func updateConfirmation(log: DetectionLog, confirmed: Bool) {
        log.userConfirmed = confirmed
        try? modelContext.save()
    }
}

#Preview {
    NavigationStack {
        PatrolMapView()
    }
}
