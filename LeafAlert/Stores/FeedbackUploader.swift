import Foundation
import Combine

/// Discovers the LeafAlert feedback server on the local network via Bonjour
/// and uploads feedback entries (image + metadata) over HTTP.
///
/// # Discovery
/// Uses `NetServiceBrowser` to find `_leafalert._tcp` services. When found,
/// resolves the hostname/port and stores the server URL.
///
/// # Upload Flow
/// After each feedback submission, `FeedbackExporter` calls `uploadPendingFeedback()`.
/// This reads the local manifest, compares against a "synced" set, and uploads
/// new entries via multipart POST.
final class FeedbackUploader: NSObject, ObservableObject {

    static let shared = FeedbackUploader()

    // MARK: - Published State

    /// Whether a sync server has been discovered on the network.
    @Published private(set) var isServerAvailable = false

    /// Human-readable server address (e.g., "192.168.1.42:8847").
    @Published private(set) var serverAddress: String?

    /// Number of entries successfully synced.
    @Published private(set) var syncedCount = 0

    /// Number of entries pending upload.
    @Published private(set) var pendingCount = 0

    /// Whether a sync is currently in progress.
    @Published private(set) var isSyncing = false

    /// Last sync timestamp.
    @Published private(set) var lastSyncDate: Date?

    /// Last error message (cleared on successful sync).
    @Published private(set) var lastError: String?

    // MARK: - Private

    private var serverURL: URL?
    private let browser = NetServiceBrowser()
    private var discoveredService: NetService?
    private var syncedFilenames: Set<String> = []
    private let session = URLSession(configuration: .ephemeral)
    private let syncQueue = DispatchQueue(label: "com.leafalert.feedback-upload")

    private override init() {
        super.init()
        loadSyncState()
        startBrowsing()
    }

    // MARK: - Bonjour Discovery

    func startBrowsing() {
        browser.delegate = self
        browser.searchForServices(ofType: "_leafalert._tcp.", inDomain: "local.")
    }

    func stopBrowsing() {
        browser.stop()
    }

    // MARK: - Manual Server Entry

    /// Set the server URL manually (e.g., from a text field in settings).
    func setManualServer(host: String, port: Int = 8847) {
        let url = URL(string: "http://\(host):\(port)")!
        serverURL = url
        DispatchQueue.main.async {
            self.serverAddress = "\(host):\(port)"
            self.isServerAvailable = true
        }
        // Verify connectivity
        pingServer(url: url)
    }

    // MARK: - Sync

    /// Upload all un-synced feedback entries to the server.
    func syncAll() {
        guard let serverURL else {
            DispatchQueue.main.async { self.lastError = "No server found" }
            return
        }
        guard !isSyncing else { return }

        DispatchQueue.main.async { self.isSyncing = true; self.lastError = nil }

        syncQueue.async { [weak self] in
            self?.performSync(to: serverURL)
        }
    }

    /// Upload a single feedback entry immediately (called after each submission).
    func uploadEntry(
        imageData: Data?,
        metadata: [String: Any],
        filename: String
    ) {
        guard let serverURL else { return }

        // Confine the de-dup check to syncQueue: syncedFilenames is not
        // thread-safe and may be mutated concurrently from upload completions.
        syncQueue.async { [weak self] in
            guard let self else { return }
            guard !self.syncedFilenames.contains(filename) else { return }
            self.uploadSingleEntry(
                to: serverURL,
                imageData: imageData,
                metadata: metadata,
                filename: filename
            )
        }
    }

    // MARK: - Private Sync Logic

    private func performSync(to url: URL) {
        let feedbackDir = FeedbackExporter.shared.feedbackDirectoryURL
        let manifestURL = feedbackDir.appendingPathComponent("manifest.json")

        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let entries = manifest["entries"] as? [[String: Any]] else {
            DispatchQueue.main.async {
                self.isSyncing = false
                self.pendingCount = 0
            }
            return
        }

        let pending = entries.filter { entry in
            guard let filename = entry["filename"] as? String else { return false }
            return !syncedFilenames.contains(filename)
        }

        DispatchQueue.main.async { self.pendingCount = pending.count }

        // successCount is mutated only on syncQueue (every completion hops
        // back to syncQueue before touching it), so no atomic is needed.
        var successCount = 0
        let group = DispatchGroup()

        for entry in pending {
            guard let filename = entry["filename"] as? String else { continue }

            group.enter()

            // Load image data
            let imageURL = feedbackDir.appendingPathComponent(filename)
            let imageData = try? Data(contentsOf: imageURL)

            uploadSingleEntry(to: url, imageData: imageData, metadata: entry, filename: filename) { [weak self] success in
                guard let self else { group.leave(); return }
                // Hop onto syncQueue before touching any shared mutable state.
                self.syncQueue.async {
                    if success {
                        // insert() returns false if already present, which guards
                        // against double-counting if the same entry somehow
                        // completes twice.
                        let inserted = self.syncedFilenames.insert(filename).inserted
                        if inserted {
                            successCount += 1
                        }
                        let syncedTotal = self.syncedFilenames.count
                        DispatchQueue.main.async {
                            self.syncedCount = syncedTotal
                            self.pendingCount = max(0, self.pendingCount - 1)
                        }
                    }
                    group.leave()
                }
            }
        }

        // Finalize on syncQueue once all uploads have completed. Using
        // notify (rather than a blocking wait) keeps syncQueue free to run
        // the per-completion hops above; blocking here would deadlock the
        // serial queue against its own pending work.
        group.notify(queue: syncQueue) {
            self.saveSyncState()

            // successCount is read on syncQueue, where it is mutated, so this
            // is a consistent snapshot; UI updates are published to main.
            let finalSuccessCount = successCount
            let total = pending.count
            DispatchQueue.main.async {
                self.isSyncing = false
                self.lastSyncDate = Date()
                if finalSuccessCount == total {
                    self.lastError = nil
                } else {
                    self.lastError = "Synced \(finalSuccessCount)/\(total)"
                }
            }
        }
    }

    private func uploadSingleEntry(
        to serverURL: URL,
        imageData: Data?,
        metadata: [String: Any],
        filename: String,
        completion: ((Bool) -> Void)? = nil
    ) {
        let uploadURL = serverURL.appendingPathComponent("upload")
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 30

        let boundary = "LeafAlert-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Metadata part
        let metadataJSON = (try? JSONSerialization.data(withJSONObject: metadata, options: [])) ?? Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"metadata\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: application/json\r\n\r\n".data(using: .utf8)!)
        body.append(metadataJSON)
        body.append("\r\n".data(using: .utf8)!)

        // Image part
        if let imageData {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let task = session.dataTask(with: request) { [weak self] _, response, error in
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            let success = error == nil && statusCode == 200

            // When a completion handler is supplied (the performSync path), it
            // owns all ledger mutation on syncQueue; doing it here too would
            // double-insert. Only manage the ledger ourselves on the standalone
            // (uploadEntry) path, and always confine that to syncQueue since the
            // URLSession completion runs on its own delegate queue.
            if let completion {
                completion(success)
            } else if success, let self {
                self.syncQueue.async {
                    let inserted = self.syncedFilenames.insert(filename).inserted
                    if inserted {
                        self.saveSyncState()
                    }
                    let syncedTotal = self.syncedFilenames.count
                    DispatchQueue.main.async {
                        self.syncedCount = syncedTotal
                    }
                }
            }
        }
        task.resume()
    }

    private func pingServer(url: URL) {
        let pingURL = url.appendingPathComponent("ping")
        var request = URLRequest(url: pingURL)
        request.timeoutInterval = 5

        session.dataTask(with: request) { [weak self] data, response, error in
            let ok = error == nil && (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async {
                self?.isServerAvailable = ok
                if !ok {
                    self?.lastError = "Server not reachable"
                }
            }
        }.resume()
    }

    // MARK: - Persistence

    private var syncStateURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("feedback_sync_state.json")
    }

    private func loadSyncState() {
        guard let data = try? Data(contentsOf: syncStateURL),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let filenames = state["syncedFilenames"] as? [String] else { return }
        syncedFilenames = Set(filenames)
        syncedCount = syncedFilenames.count
    }

    private func saveSyncState() {
        let state: [String: Any] = ["syncedFilenames": Array(syncedFilenames)]
        if let data = try? JSONSerialization.data(withJSONObject: state) {
            try? data.write(to: syncStateURL, options: .atomic)
        }
    }
}

// MARK: - NetServiceBrowserDelegate

extension FeedbackUploader: NetServiceBrowserDelegate {

    func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        guard service.type == "_leafalert._tcp." else { return }
        discoveredService = service
        service.delegate = self
        service.resolve(withTimeout: 10)
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool) {
        if service == discoveredService {
            discoveredService = nil
            DispatchQueue.main.async {
                self.isServerAvailable = false
                self.serverAddress = nil
                self.serverURL = nil
            }
        }
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didNotSearch errorDict: [String: NSNumber]) {
        print("[FeedbackUploader] Bonjour search failed: \(errorDict)")
    }
}

// MARK: - NetServiceDelegate

extension FeedbackUploader: NetServiceDelegate {

    func netServiceDidResolveAddress(_ sender: NetService) {
        guard let addresses = sender.addresses, !addresses.isEmpty else { return }

        // Extract IP from resolved addresses
        var hostname = sender.hostName ?? "unknown"
        if hostname.hasSuffix(".") {
            hostname = String(hostname.dropLast())
        }

        let port = sender.port
        let url = URL(string: "http://\(hostname):\(port)")!

        serverURL = url
        DispatchQueue.main.async {
            self.serverAddress = "\(hostname):\(port)"
            self.isServerAvailable = true
        }

        print("[FeedbackUploader] Discovered server at \(hostname):\(port)")
        pingServer(url: url)
    }

    func netService(_ sender: NetService, didNotResolve errorDict: [String: NSNumber]) {
        print("[FeedbackUploader] Failed to resolve service: \(errorDict)")
    }
}
