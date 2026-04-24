const std = @import("std");
const builtin = @import("builtin");

const MAX_SESSIONS: usize = 500;
const SESSION_TTL: f64 = 21600.0;
const MAX_TOOL_ITERATIONS: usize = 20;
const MAX_HISTORY_MESSAGES: usize = 60;
const MAX_STORED_MESSAGES: usize = 120;
const MAX_SUMMARY_CHARS: usize = 600;
const EXA_NUM_RESULTS: usize = 100;
const MAX_EXA_CONCURRENT: usize = 5;
const MAX_MODEL_CONTENT_CHARS: usize = 180000;
const MAX_DELETED_SESSIONS: usize = 1000;
const MODEL_CONTEXT_LIMIT: usize = 202752;
const DESIRED_MAX_TOKENS: usize = 180000;
const MAX_MESSAGE_LENGTH: usize = 100000;
const MAX_PROMPT_LENGTH: usize = 10000;
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_INLINE_IMAGE_BYTES: usize = 512 * 1024;
const EXA_QPS_LIMIT: usize = 8;
const MAX_REFERENCE_IMAGES: usize = 13;

const ExaToolResult = struct {
    events: std.ArrayList([]u8),
    tool_msg: Message,
    tool_type: []const u8,
};

const AgentToolResult = struct {
    result: []u8,
    events: std.ArrayList([]u8),
};
const IMAGE_REF_DELIM = "|||";
const SESSION_PING_INTERVAL_MS: u64 = 5000;

const ALLOWED_IMAGE_MIMES = [_][]const u8{
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
};

const DEEP_SEARCH_TYPES = [_][]const u8{ "deep", "deep-reasoning" };

const MimeExt = struct {
    mime: []const u8,
    ext: []const u8,
};

const MIME_TO_EXT = [_]MimeExt{
    .{ .mime = "image/jpeg", .ext = ".jpg" },
    .{ .mime = "image/png", .ext = ".png" },
    .{ .mime = "image/webp", .ext = ".webp" },
    .{ .mime = "image/gif", .ext = ".gif" },
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var global_allocator: std.mem.Allocator = undefined;

var hpc_ai_api_key: []u8 = &[_]u8{};
var exa_api_key: []u8 = &[_]u8{};
var fireworks_api_key: []u8 = &[_]u8{};
var obj_bucket: []u8 = &[_]u8{};
var project_root: []u8 = &[_]u8{};
var index_file: []u8 = &[_]u8{};
var static_dir: []u8 = &[_]u8{};
var static_sw_file: []u8 = &[_]u8{};
var persistence_file: []u8 = &[_]u8{};
var server_host: []u8 = &[_]u8{};
var server_port: u16 = 5000;

const MessageContentPart = struct {
    type: []u8,
    text: ?[]u8,
    media_type: ?[]u8,
    data: ?[]u8,
    key: ?[]u8,
    image_url: ?ImageUrl,

    const ImageUrl = struct {
        url: []u8,
        detail: []u8,
    };

    fn deinit(self: *MessageContentPart, allocator: std.mem.Allocator) void {
        allocator.free(self.type);
        if (self.text) |t| allocator.free(t);
        if (self.media_type) |m| allocator.free(m);
        if (self.data) |d| allocator.free(d);
        if (self.key) |k| allocator.free(k);
        if (self.image_url) |*iu| {
            allocator.free(iu.url);
            allocator.free(iu.detail);
        }
    }

    fn clone(self: MessageContentPart, allocator: std.mem.Allocator) !MessageContentPart {
        const result: MessageContentPart = .{
            .type = try allocator.dupe(u8, self.type),
            .text = if (self.text) |t| try allocator.dupe(u8, t) else null,
            .media_type = if (self.media_type) |m| try allocator.dupe(u8, m) else null,
            .data = if (self.data) |d| try allocator.dupe(u8, d) else null,
            .key = if (self.key) |k| try allocator.dupe(u8, k) else null,
            .image_url = if (self.image_url) |iu| ImageUrl{
                .url = try allocator.dupe(u8, iu.url),
                .detail = try allocator.dupe(u8, iu.detail),
            } else null,
        };
        return result;
    }
};

const ToolCallFunction = struct {
    name: []u8,
    arguments: []u8,

    fn deinit(self: *ToolCallFunction, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.arguments);
    }

    fn clone(self: ToolCallFunction, allocator: std.mem.Allocator) !ToolCallFunction {
        return ToolCallFunction{
            .name = try allocator.dupe(u8, self.name),
            .arguments = try allocator.dupe(u8, self.arguments),
        };
    }
};

const ToolCall = struct {
    id: []u8,
    type: []u8,
    function: ToolCallFunction,

    fn deinit(self: *ToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.type);
        self.function.deinit(allocator);
    }

    fn clone(self: ToolCall, allocator: std.mem.Allocator) !ToolCall {
        return ToolCall{
            .id = try allocator.dupe(u8, self.id),
            .type = try allocator.dupe(u8, self.type),
            .function = try self.function.clone(allocator),
        };
    }
};

const MessageContent = union(enum) {
    text: []u8,
    parts: std.ArrayList(MessageContentPart),

    fn deinit(self: *MessageContent, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .text => |t| allocator.free(t),
            .parts => |*p| {
                for (p.items) |*part| {
                    part.deinit(allocator);
                }
                p.deinit();
            },
        }
    }

    fn clone(self: MessageContent, allocator: std.mem.Allocator) !MessageContent {
        switch (self) {
            .text => |t| return MessageContent{ .text = try allocator.dupe(u8, t) },
            .parts => |p| {
                var new_parts = std.ArrayList(MessageContentPart).init(allocator);
                for (p.items) |part| {
                    try new_parts.append(try part.clone(allocator));
                }
                return MessageContent{ .parts = new_parts };
            },
        }
    }
};

const Message = struct {
    role: []u8,
    content: MessageContent,
    tool_calls: ?std.ArrayList(ToolCall),
    tool_call_id: ?[]u8,
    msg_id: ?[]u8,
    agent_mode: bool,
    cached_size: ?usize,

    fn deinit(self: *Message, allocator: std.mem.Allocator) void {
        allocator.free(self.role);
        self.content.deinit(allocator);
        if (self.tool_calls) |*tcs| {
            for (tcs.items) |*tc| {
                tc.deinit(allocator);
            }
            tcs.deinit();
        }
        if (self.tool_call_id) |id| allocator.free(id);
        if (self.msg_id) |mid| allocator.free(mid);
    }

    fn clone(self: Message, allocator: std.mem.Allocator) !Message {
        var result: Message = .{
            .role = try allocator.dupe(u8, self.role),
            .content = try self.content.clone(allocator),
            .tool_calls = null,
            .tool_call_id = if (self.tool_call_id) |id| try allocator.dupe(u8, id) else null,
            .msg_id = if (self.msg_id) |mid| try allocator.dupe(u8, mid) else null,
            .agent_mode = self.agent_mode,
            .cached_size = self.cached_size,
        };
        if (self.tool_calls) |tcs| {
            var new_tcs = std.ArrayList(ToolCall).init(allocator);
            for (tcs.items) |tc| {
                try new_tcs.append(try tc.clone(allocator));
            }
            result.tool_calls = new_tcs;
        }
        return result;
    }
};

const Session = struct {
    messages: std.ArrayList(Message),
    created_at: f64,
    updated_at: f64,

    fn init(allocator: std.mem.Allocator, created_at: f64) Session {
        return Session{
            .messages = std.ArrayList(Message).init(allocator),
            .created_at = created_at,
            .updated_at = created_at,
        };
    }

    fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        for (self.messages.items) |*msg| {
            msg.deinit(allocator);
        }
        self.messages.deinit();
    }
};

const RateLimiter = struct {
    max: usize,
    timestamps: std.ArrayList(i64),
    mutex: std.Thread.Mutex,

    fn init(allocator: std.mem.Allocator, max_per_second: usize) RateLimiter {
        return RateLimiter{
            .max = max_per_second,
            .timestamps = std.ArrayList(i64).init(allocator),
            .mutex = .{},
        };
    }

    fn deinit(self: *RateLimiter) void {
        self.timestamps.deinit();
    }

    fn acquire(self: *RateLimiter) void {
        while (true) {
            const now_ns = std.time.nanoTimestamp();
            const now_ms: i64 = @intCast(@divTrunc(now_ns, 1_000_000));
            const window_ms: i64 = 1000;
            {
                self.mutex.lock();
                defer self.mutex.unlock();
                var i: usize = 0;
                while (i < self.timestamps.items.len) {
                    if (now_ms - self.timestamps.items[i] >= window_ms) {
                        _ = self.timestamps.orderedRemove(i);
                    } else {
                        i += 1;
                    }
                }
                if (self.timestamps.items.len < self.max) {
                    self.timestamps.append(now_ms) catch {};
                    return;
                }
                const wait_until = self.timestamps.items[0] + window_ms + 50;
                const wait_ms = @max(0, wait_until - now_ms);
                if (wait_ms <= 0) continue;
                self.mutex.unlock();
                std.time.sleep(@intCast(wait_ms * 1_000_000));
                self.mutex.lock();
            }
        }
    }
};

var sessions_map: std.StringHashMap(Session) = undefined;
var sessions_mutex: std.Thread.Mutex = .{};

var session_locks_map: std.StringHashMap(*std.Thread.Mutex) = undefined;
var session_locks_guard: std.Thread.Mutex = .{};

var deleted_sessions_map: std.StringHashMap(f64) = undefined;
var deleted_sessions_mutex: std.Thread.Mutex = .{};

var exa_limiter: RateLimiter = undefined;
var session_exa_limiters_map: std.StringHashMap(*RateLimiter) = undefined;
var session_exa_limiters_mutex: std.Thread.Mutex = .{};

fn getEnvAlloc(allocator: std.mem.Allocator, key: []const u8, default: []const u8) ![]u8 {
    const val = std.process.getEnvVarOwned(allocator, key) catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            return allocator.dupe(u8, default);
        }
        return err;
    };
    return val;
}

fn nowSeconds() f64 {
    return @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;
}

fn generateUuid(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const result = try std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        bytes[0],  bytes[1],  bytes[2],  bytes[3],
        bytes[4],  bytes[5],
        bytes[6],  bytes[7],
        bytes[8],  bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    });
    return result;
}

fn generateShortHex(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [4]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    return std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{ bytes[0], bytes[1], bytes[2], bytes[3] });
}

fn detectImageFormat(data: []const u8) struct { mime: []const u8, ext: []const u8 } {
    if (data.len >= 8 and std.mem.eql(u8, data[0..8], "\x89PNG\r\n\x1a\n")) {
        return .{ .mime = "image/png", .ext = ".png" };
    }
    if (data.len >= 2 and data[0] == 0xff and data[1] == 0xd8) {
        return .{ .mime = "image/jpeg", .ext = ".jpg" };
    }
    if (data.len >= 12 and std.mem.eql(u8, data[0..4], "RIFF") and std.mem.eql(u8, data[8..12], "WEBP")) {
        return .{ .mime = "image/webp", .ext = ".webp" };
    }
    if (data.len >= 6 and (std.mem.eql(u8, data[0..6], "GIF87a") or std.mem.eql(u8, data[0..6], "GIF89a"))) {
        return .{ .mime = "image/gif", .ext = ".gif" };
    }
    return .{ .mime = "application/octet-stream", .ext = "" };
}

fn isMimeAllowed(mime: []const u8) bool {
    for (ALLOWED_IMAGE_MIMES) |allowed| {
        if (std.mem.eql(u8, mime, allowed)) return true;
    }
    return false;
}

fn isDeepSearchType(t: []const u8) bool {
    for (DEEP_SEARCH_TYPES) |ds| {
        if (std.mem.eql(u8, t, ds)) return true;
    }
    return false;
}

fn mimeToExt(mime: []const u8) []const u8 {
    for (MIME_TO_EXT) |me| {
        if (std.mem.eql(u8, me.mime, mime)) return me.ext;
    }
    return ".img";
}

fn extToMime(ext: []const u8) ?[]const u8 {
    for (MIME_TO_EXT) |me| {
        if (std.mem.eql(u8, me.ext, ext)) return me.mime;
    }
    return null;
}

fn stripWhitespace(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = try std.ArrayList(u8).initCapacity(allocator, s.len);
    defer result.deinit();
    for (s) |c| {
        if (c != ' ' and c != '\t' and c != '\n' and c != '\r') {
            try result.append(c);
        }
    }
    return result.toOwnedSlice();
}

fn safeB64Decode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const stripped = try stripWhitespace(allocator, s);
    defer allocator.free(stripped);

    const pad = 4 - (stripped.len % 4);
    var padded: []u8 = stripped;
    var padded_owned = false;
    if (pad != 4) {
        padded = try allocator.alloc(u8, stripped.len + pad);
        padded_owned = true;
        @memcpy(padded[0..stripped.len], stripped);
        for (stripped.len..padded.len) |i| {
            padded[i] = '=';
        }
    }
    defer if (padded_owned) allocator.free(padded);

    const is_url_safe = std.mem.indexOfAny(u8, padded, "-_") != null;

    if (is_url_safe) {
        const converted = try allocator.dupe(u8, padded);
        defer allocator.free(converted);
        for (converted) |*c| {
            if (c.* == '-') c.* = '+';
            if (c.* == '_') c.* = '/';
        }
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(converted) catch return error.InvalidBase64;
        const decoded = try allocator.alloc(u8, decoded_size);
        std.base64.standard.Decoder.decode(decoded, converted) catch {
            allocator.free(decoded);
            return error.InvalidBase64;
        };
        return decoded;
    } else {
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(padded) catch return error.InvalidBase64;
        const decoded = try allocator.alloc(u8, decoded_size);
        std.base64.standard.Decoder.decode(decoded, padded) catch {
            allocator.free(decoded);
            return error.InvalidBase64;
        };
        return decoded;
    }
}

fn decodeAndStripB64(allocator: std.mem.Allocator, raw: []const u8) !struct { stripped: []u8, decoded: []u8 } {
    if (raw.len == 0) return error.EmptyBase64;
    var working: []const u8 = raw;
    if (std.mem.indexOf(u8, working, ",")) |comma_idx| {
        working = working[comma_idx + 1 ..];
    }
    const stripped = try stripWhitespace(allocator, working);
    if (stripped.len == 0) {
        allocator.free(stripped);
        return error.EmptyBase64AfterStrip;
    }
    const decoded = safeB64Decode(allocator, stripped) catch {
        allocator.free(stripped);
        return error.InvalidBase64Encoding;
    };
    return .{ .stripped = stripped, .decoded = decoded };
}

fn normalizeInputImage(allocator: std.mem.Allocator, data: []const u8, media_type: []const u8) !struct {
    data: []u8,
    bytes: []u8,
    media_type: []u8,
} {
    const result = try decodeAndStripB64(allocator, data);
    if (result.decoded.len > MAX_IMAGE_BYTES) {
        allocator.free(result.stripped);
        allocator.free(result.decoded);
        return error.ImageTooLarge;
    }
    const detected = detectImageFormat(result.decoded);
    if (!isMimeAllowed(detected.mime)) {
        allocator.free(result.stripped);
        allocator.free(result.decoded);
        return error.UnsupportedImageFormat;
    }
    if (!std.mem.eql(u8, detected.mime, media_type)) {
        allocator.free(result.stripped);
        allocator.free(result.decoded);
        return error.ImageMediaTypeMismatch;
    }
    return .{
        .data = result.stripped,
        .bytes = result.decoded,
        .media_type = try allocator.dupe(u8, detected.mime),
    };
}

fn pathInProject(full_path: []const u8) bool {
    return std.mem.eql(u8, full_path, project_root) or std.mem.startsWith(u8, full_path, project_root);
}

fn pruneDeletedSessionsUnlocked(now: f64) void {
    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();
    var it = deleted_sessions_map.iterator();
    while (it.next()) |entry| {
        if (now - entry.value_ptr.* > SESSION_TTL) {
            to_remove.append(entry.key_ptr.*) catch {};
        }
    }
    for (to_remove.items) |key| {
        if (deleted_sessions_map.fetchRemove(key)) |kv| {
            global_allocator.free(kv.key);
        }
    }
    if (deleted_sessions_map.count() > MAX_DELETED_SESSIONS) {
        const DeletedEntry = struct { key: []const u8, val: f64 };
        var entries = std.ArrayList(DeletedEntry).init(global_allocator);
        defer entries.deinit();
        var it2 = deleted_sessions_map.iterator();
        while (it2.next()) |entry| {
            entries.append(.{ .key = entry.key_ptr.*, .val = entry.value_ptr.* }) catch {};
        }
        std.sort.block(DeletedEntry, entries.items, {}, struct {
            fn lessThan(_: void, a: DeletedEntry, b: DeletedEntry) bool {
                return a.val < b.val;
            }
        }.lessThan);
        const excess = deleted_sessions_map.count() - MAX_DELETED_SESSIONS;
        for (entries.items[0..excess]) |e| {
            if (deleted_sessions_map.fetchRemove(e.key)) |kv| {
                global_allocator.free(kv.key);
            }
        }
    }
}

fn isDeleted(sid: []const u8) bool {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    pruneDeletedSessionsUnlocked(nowSeconds());
    const deleted_at = deleted_sessions_map.get(sid) orelse return false;
    if (nowSeconds() - deleted_at > SESSION_TTL) return false;
    return true;
}

fn markDeleted(sid: []const u8) void {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    pruneDeletedSessionsUnlocked(nowSeconds());
    const key = global_allocator.dupe(u8, sid) catch return;
    deleted_sessions_map.put(key, nowSeconds()) catch {
        global_allocator.free(key);
    };
}

fn unmarkDeleted(sid: []const u8) void {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    if (deleted_sessions_map.fetchRemove(sid)) |kv| {
        global_allocator.free(kv.key);
    }
    pruneDeletedSessionsUnlocked(nowSeconds());
}

fn getOrCreateSessionLock(sid: []const u8) !*std.Thread.Mutex {
    session_locks_guard.lock();
    defer session_locks_guard.unlock();
    if (session_locks_map.get(sid)) |lock| {
        return lock;
    }
    const lock = try global_allocator.create(std.Thread.Mutex);
    lock.* = .{};
    const key = try global_allocator.dupe(u8, sid);
    try session_locks_map.put(key, lock);
    return lock;
}

fn getOrCreateExaLimiter(sid: []const u8) !*RateLimiter {
    session_exa_limiters_mutex.lock();
    defer session_exa_limiters_mutex.unlock();
    if (session_exa_limiters_map.get(sid)) |limiter| {
        return limiter;
    }
    const limiter = try global_allocator.create(RateLimiter);
    limiter.* = RateLimiter.init(global_allocator, 2);
    const key = try global_allocator.dupe(u8, sid);
    try session_exa_limiters_map.put(key, limiter);
    return limiter;
}

fn evictSessions() void {
    const now = nowSeconds();
    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();

    var locked_sids = std.StringHashMap(void).init(global_allocator);
    defer locked_sids.deinit();

    {
        session_locks_guard.lock();
        defer session_locks_guard.unlock();
        var it = session_locks_map.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*.tryLock()) {
                entry.value_ptr.*.unlock();
            } else {
                locked_sids.put(entry.key_ptr.*, {}) catch {};
            }
        }
    }

    sessions_mutex.lock();
    defer sessions_mutex.unlock();

    var it = sessions_map.iterator();
    while (it.next()) |entry| {
        if (now - entry.value_ptr.*.updated_at > SESSION_TTL) {
            if (locked_sids.get(entry.key_ptr.*) == null) {
                to_remove.append(entry.key_ptr.*) catch {};
            }
        }
    }
    for (to_remove.items) |key| {
        if (sessions_map.fetchRemove(key)) |kv| {
            var sess = kv.value;
            sess.deinit(global_allocator);
            global_allocator.free(kv.key);
        }
    }
    to_remove.clearRetainingCapacity();

    if (sessions_map.count() > MAX_SESSIONS) {
        const SessionEntry = struct { key: []const u8, updated_at: f64 };
        var entries = std.ArrayList(SessionEntry).init(global_allocator);
        defer entries.deinit();
        var it2 = sessions_map.iterator();
        while (it2.next()) |entry| {
            if (locked_sids.get(entry.key_ptr.*) == null) {
                entries.append(.{ .key = entry.key_ptr.*, .updated_at = entry.value_ptr.*.updated_at }) catch {};
            }
        }
        std.sort.block(SessionEntry, entries.items, {}, struct {
            fn lessThan(_: void, a: SessionEntry, b: SessionEntry) bool {
                return a.updated_at < b.updated_at;
            }
        }.lessThan);
        const overflow = sessions_map.count() - MAX_SESSIONS;
        for (entries.items[0..@min(overflow, entries.items.len)]) |e| {
            if (sessions_map.fetchRemove(e.key)) |kv| {
                var sess = kv.value;
                sess.deinit(global_allocator);
                global_allocator.free(kv.key);
            }
        }
    }
}

fn saveSessionsToDisk() void {
    var json_buf = std.ArrayList(u8).init(global_allocator);
    defer json_buf.deinit();

    sessions_mutex.lock();
    const snap_keys = blk: {
        var keys = std.ArrayList([]const u8).init(global_allocator);
        var it = sessions_map.keyIterator();
        while (it.next()) |k| {
            keys.append(k.*) catch {};
        }
        break :blk keys;
    };
    defer snap_keys.deinit();

    var snap_sessions = std.ArrayList(struct { key: []const u8, sess: *Session }).init(global_allocator);
    defer snap_sessions.deinit();
    for (snap_keys.items) |k| {
        if (sessions_map.getPtr(k)) |s| {
            snap_sessions.append(.{ .key = k, .sess = s }) catch {};
        }
    }
    sessions_mutex.unlock();

    var writer = json_buf.writer();
    writer.writeByte('{') catch return;
    var first_sess = true;
    for (snap_sessions.items) |entry| {
        if (!first_sess) writer.writeByte(',') catch return;
        first_sess = false;
        writeJsonString(writer, entry.key) catch return;
        writer.writeByte(':') catch return;
        writer.writeByte('{') catch return;
        writer.writeAll("\"messages\":") catch return;
        serializeMessages(writer, entry.sess.messages.items) catch return;
        writer.writeAll(",\"created_at\":") catch return;
        std.fmt.format(writer, "{d}", .{entry.sess.created_at}) catch return;
        writer.writeAll(",\"updated_at\":") catch return;
        std.fmt.format(writer, "{d}", .{entry.sess.updated_at}) catch return;
        writer.writeByte('}') catch return;
    }
    writer.writeByte('}') catch return;

    const tmp_file = std.fmt.allocPrint(global_allocator, "{s}.tmp", .{persistence_file}) catch return;
    defer global_allocator.free(tmp_file);

    const f = std.fs.createFileAbsolute(tmp_file, .{}) catch return;
    f.writeAll(json_buf.items) catch {
        f.close();
        return;
    };
    f.close();

    std.fs.renameAbsolute(tmp_file, persistence_file) catch {};
}

fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => {
                try std.fmt.format(writer, "\\u{x:0>4}", .{c});
            },
            else => try writer.writeByte(c),
        }
    }
    try writer.writeByte('"');
}

fn serializeMessages(writer: anytype, messages: []const Message) !void {
    try writer.writeByte('[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try serializeMessage(writer, msg);
    }
    try writer.writeByte(']');
}

fn serializeMessage(writer: anytype, msg: Message) !void {
    try writer.writeByte('{');
    try writer.writeAll("\"role\":");
    try writeJsonString(writer, msg.role);
    try writer.writeAll(",\"content\":");
    switch (msg.content) {
        .text => |t| try writeJsonString(writer, t),
        .parts => |parts| {
            try writer.writeByte('[');
            for (parts.items, 0..) |part, i| {
                if (i > 0) try writer.writeByte(',');
                try serializeContentPart(writer, part);
            }
            try writer.writeByte(']');
        },
    }
    if (msg.tool_calls) |tcs| {
        try writer.writeAll(",\"tool_calls\":[");
        for (tcs.items, 0..) |tc, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.writeAll("{\"id\":");
            try writeJsonString(writer, tc.id);
            try writer.writeAll(",\"type\":");
            try writeJsonString(writer, tc.type);
            try writer.writeAll(",\"function\":{\"name\":");
            try writeJsonString(writer, tc.function.name);
            try writer.writeAll(",\"arguments\":");
            try writeJsonString(writer, tc.function.arguments);
            try writer.writeAll("}}");
        }
        try writer.writeByte(']');
    }
    if (msg.tool_call_id) |id| {
        try writer.writeAll(",\"tool_call_id\":");
        try writeJsonString(writer, id);
    }
    if (msg.msg_id) |mid| {
        try writer.writeAll(",\"msg_id\":");
        try writeJsonString(writer, mid);
    }
    if (msg.agent_mode) {
        try writer.writeAll(",\"agentMode\":true");
    }
    try writer.writeByte('}');
}

fn serializeContentPart(writer: anytype, part: MessageContentPart) !void {
    try writer.writeByte('{');
    try writer.writeAll("\"type\":");
    try writeJsonString(writer, part.type);
    if (part.text) |t| {
        try writer.writeAll(",\"text\":");
        try writeJsonString(writer, t);
    }
    if (part.media_type) |m| {
        try writer.writeAll(",\"media_type\":");
        try writeJsonString(writer, m);
    }
    if (part.data) |d| {
        try writer.writeAll(",\"data\":");
        try writeJsonString(writer, d);
    }
    if (part.key) |k| {
        try writer.writeAll(",\"key\":");
        try writeJsonString(writer, k);
    }
    if (part.image_url) |iu| {
        try writer.writeAll(",\"image_url\":{\"url\":");
        try writeJsonString(writer, iu.url);
        try writer.writeAll(",\"detail\":");
        try writeJsonString(writer, iu.detail);
        try writer.writeByte('}');
    }
    try writer.writeByte('}');
}

fn loadSessionsFromDisk() void {
    const f = std.fs.openFileAbsolute(persistence_file, .{}) catch return;
    defer f.close();
    const content = f.readToEndAlloc(global_allocator, 100 * 1024 * 1024) catch return;
    defer global_allocator.free(content);

    const parsed = std.json.parseFromSlice(std.json.Value, global_allocator, content, .{}) catch return;
    defer parsed.deinit();

    if (parsed.value != .object) return;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const k = entry.key_ptr.*;
        const v = entry.value_ptr.*;
        if (v != .object) continue;
        const msgs_val = v.object.get("messages") orelse continue;
        const updated_val = v.object.get("updated_at") orelse continue;
        const created_val = v.object.get("created_at") orelse v.object.get("updated_at") orelse continue;
        if (msgs_val != .array) continue;

        const created_at = jsonValueToFloat(created_val);
        const updated_at = jsonValueToFloat(updated_val);

        var sess = Session.init(global_allocator, created_at);
        sess.updated_at = updated_at;

        for (msgs_val.array.items) |msg_val| {
            if (msg_val != .object) continue;
            const msg = parseMessageFromJson(msg_val) catch continue;
            sess.messages.append(msg) catch {
                var m = msg;
                m.deinit(global_allocator);
                continue;
            };
        }

        const key = global_allocator.dupe(u8, k) catch {
            sess.deinit(global_allocator);
            continue;
        };
        sessions_mutex.lock();
        sessions_map.put(key, sess) catch {
            sessions_mutex.unlock();
            sess.deinit(global_allocator);
            global_allocator.free(key);
            continue;
        };
        sessions_mutex.unlock();
    }
}

fn jsonValueToFloat(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0.0,
        else => 0.0,
    };
}

fn jsonValueToString(allocator: std.mem.Allocator, v: std.json.Value) ![]u8 {
    return switch (v) {
        .string => |s| allocator.dupe(u8, s),
        .integer => |i| std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| std.fmt.allocPrint(allocator, "{d}", .{f}),
        .bool => |b| allocator.dupe(u8, if (b) "true" else "false"),
        .null => allocator.dupe(u8, "null"),
        else => allocator.dupe(u8, ""),
    };
}

fn parseMessageFromJson(v: std.json.Value) !Message {
    const obj = v.object;
    const role_val = obj.get("role") orelse return error.MissingRole;
    const role = try global_allocator.dupe(u8, if (role_val == .string) role_val.string else "user");

    const content_val = obj.get("content") orelse std.json.Value{ .string = "" };
    var content: MessageContent = undefined;

    if (content_val == .string) {
        content = MessageContent{ .text = try global_allocator.dupe(u8, content_val.string) };
    } else if (content_val == .array) {
        var parts = std.ArrayList(MessageContentPart).init(global_allocator);
        for (content_val.array.items) |part_val| {
            if (part_val != .object) continue;
            const part = try parseContentPartFromJson(part_val);
            try parts.append(part);
        }
        content = MessageContent{ .parts = parts };
    } else {
        content = MessageContent{ .text = try global_allocator.dupe(u8, "") };
    }

    var tool_calls: ?std.ArrayList(ToolCall) = null;
    if (obj.get("tool_calls")) |tcs_val| {
        if (tcs_val == .array) {
            var tcs = std.ArrayList(ToolCall).init(global_allocator);
            for (tcs_val.array.items) |tc_val| {
                if (tc_val != .object) continue;
                const tc = try parseToolCallFromJson(tc_val);
                try tcs.append(tc);
            }
            tool_calls = tcs;
        }
    }

    const tool_call_id = if (obj.get("tool_call_id")) |id_val|
        if (id_val == .string) try global_allocator.dupe(u8, id_val.string) else null
    else
        null;

    const msg_id = if (obj.get("msg_id")) |mid_val|
        if (mid_val == .string) try global_allocator.dupe(u8, mid_val.string) else null
    else
        null;

    const agent_mode = if (obj.get("agentMode")) |am_val|
        if (am_val == .bool) am_val.bool else false
    else
        false;

    return Message{
        .role = role,
        .content = content,
        .tool_calls = tool_calls,
        .tool_call_id = tool_call_id,
        .msg_id = msg_id,
        .agent_mode = agent_mode,
        .cached_size = null,
    };
}

fn parseContentPartFromJson(v: std.json.Value) !MessageContentPart {
    const obj = v.object;
    const type_val = obj.get("type") orelse return error.MissingType;
    const part_type = try global_allocator.dupe(u8, if (type_val == .string) type_val.string else "text");

    const text = if (obj.get("text")) |t|
        if (t == .string) try global_allocator.dupe(u8, t.string) else null
    else
        null;

    const media_type = if (obj.get("media_type")) |m|
        if (m == .string) try global_allocator.dupe(u8, m.string) else null
    else
        null;

    const data = if (obj.get("data")) |d|
        if (d == .string) try global_allocator.dupe(u8, d.string) else null
    else
        null;

    const key = if (obj.get("key")) |k|
        if (k == .string) try global_allocator.dupe(u8, k.string) else null
    else
        null;

    var image_url: ?MessageContentPart.ImageUrl = null;
    if (obj.get("image_url")) |iu_val| {
        if (iu_val == .object) {
            const url_val = iu_val.object.get("url") orelse std.json.Value{ .string = "" };
            const detail_val = iu_val.object.get("detail") orelse std.json.Value{ .string = "" };
            image_url = .{
                .url = try global_allocator.dupe(u8, if (url_val == .string) url_val.string else ""),
                .detail = try global_allocator.dupe(u8, if (detail_val == .string) detail_val.string else ""),
            };
        }
    }

    return MessageContentPart{
        .type = part_type,
        .text = text,
        .media_type = media_type,
        .data = data,
        .key = key,
        .image_url = image_url,
    };
}

fn parseToolCallFromJson(v: std.json.Value) !ToolCall {
    const obj = v.object;
    const id_val = obj.get("id") orelse std.json.Value{ .string = "" };
    const type_val = obj.get("type") orelse std.json.Value{ .string = "function" };
    const func_val = obj.get("function") orelse std.json.Value{ .object = std.json.ObjectMap.init(global_allocator) };

    var name: []u8 = try global_allocator.dupe(u8, "");
    var arguments: []u8 = try global_allocator.dupe(u8, "{}");

    if (func_val == .object) {
        if (func_val.object.get("name")) |n| {
            if (n == .string) {
                global_allocator.free(name);
                name = try global_allocator.dupe(u8, n.string);
            }
        }
        if (func_val.object.get("arguments")) |a| {
            if (a == .string) {
                global_allocator.free(arguments);
                arguments = try global_allocator.dupe(u8, a.string);
            }
        }
    }

    return ToolCall{
        .id = try global_allocator.dupe(u8, if (id_val == .string) id_val.string else ""),
        .type = try global_allocator.dupe(u8, if (type_val == .string) type_val.string else "function"),
        .function = .{
            .name = name,
            .arguments = arguments,
        },
    };
}

fn messageCharSize(msg: *Message) usize {
    if (msg.cached_size) |cs| return cs;
    var total: usize = msg.role.len + 32;
    switch (msg.content) {
        .text => |t| total += t.len,
        .parts => |parts| {
            for (parts.items) |part| {
                const ptype = part.type;
                if (std.mem.eql(u8, ptype, "text")) {
                    total += if (part.text) |t| t.len else 0;
                } else if (std.mem.eql(u8, ptype, "image_url")) {
                    if (part.image_url) |iu| {
                        total += iu.url.len + iu.detail.len + 32;
                    } else {
                        total += 32;
                    }
                } else if (std.mem.eql(u8, ptype, "image_ref")) {
                    total += (if (part.key) |k| k.len else 0) + (if (part.media_type) |m| m.len else 0) + 32;
                } else if (std.mem.eql(u8, ptype, "image_inline")) {
                    total += (if (part.data) |d| d.len else 0) + (if (part.media_type) |m| m.len else 0) + 32;
                } else {
                    total += 64;
                }
            }
        },
    }
    if (msg.tool_calls) |tcs| {
        for (tcs.items) |tc| {
            total += tc.id.len + tc.type.len + tc.function.name.len + tc.function.arguments.len + 32;
        }
    }
    if (msg.tool_call_id) |id| total += id.len;
    msg.cached_size = total;
    return total;
}

fn safeMaxTokens(messages: []Message) usize {
    var total_chars: usize = 0;
    for (messages) |*msg| {
        total_chars += messageCharSize(@constCast(msg));
    }
    const estimated_input_tokens = (total_chars / 3) + 2000;
    const safe = if (MODEL_CONTEXT_LIMIT > estimated_input_tokens)
        @min(DESIRED_MAX_TOKENS, MODEL_CONTEXT_LIMIT - estimated_input_tokens)
    else
        1000;
    return @max(1000, safe);
}

fn findToolBoundary(messages: []const Message, start_idx: usize) usize {
    var idx = start_idx;
    while (idx < messages.len) {
        const m = messages[idx];
        if (std.mem.eql(u8, m.role, "tool")) {
            idx += 1;
            continue;
        }
        if (std.mem.eql(u8, m.role, "assistant") and m.tool_calls != null and idx + 1 < messages.len) {
            var j = idx + 1;
            while (j < messages.len and std.mem.eql(u8, messages[j].role, "tool")) {
                j += 1;
            }
            if (j > idx + 1) {
                idx = j;
                continue;
            }
        }
        break;
    }
    return idx;
}

fn findTurnBoundary(messages: []const Message, min_idx: usize) usize {
    var idx = findToolBoundary(messages, min_idx);
    while (idx < messages.len and !std.mem.eql(u8, messages[idx].role, "user")) {
        idx += 1;
    }
    if (idx >= messages.len and messages.len > 0) {
        var fallback = findToolBoundary(messages, min_idx);
        while (fallback < messages.len and std.mem.eql(u8, messages[fallback].role, "tool")) {
            fallback += 1;
        }
        idx = @min(fallback, messages.len);
    }
    return @min(idx, messages.len);
}

fn stripOrphanedAssistantToolCalls(allocator: std.mem.Allocator, messages: *std.ArrayList(Message)) !void {
    if (messages.items.len == 0) return;

    var tool_ids_present = std.StringHashMap(void).init(allocator);
    defer tool_ids_present.deinit();

    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) {
            if (msg.tool_call_id) |id| {
                try tool_ids_present.put(id, {});
            }
        }
    }

    var cleaned = std.ArrayList(Message).init(allocator);
    for (messages.items) |*msg| {
        if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var remaining_tcs = std.ArrayList(ToolCall).init(allocator);
            for (msg.tool_calls.?.items) |tc| {
                if (tool_ids_present.get(tc.id) != null) {
                    try remaining_tcs.append(try tc.clone(allocator));
                }
            }
            if (remaining_tcs.items.len > 0) {
                var new_msg = try msg.clone(allocator);
                if (new_msg.tool_calls) |*old_tcs| {
                    for (old_tcs.items) |*tc| tc.deinit(allocator);
                    old_tcs.deinit();
                }
                new_msg.tool_calls = remaining_tcs;
                try cleaned.append(new_msg);
            } else {
                remaining_tcs.deinit();
                const content_text = switch (msg.content) {
                    .text => |t| t,
                    .parts => "",
                };
                if (content_text.len > 0) {
                    const new_msg = Message{
                        .role = try allocator.dupe(u8, msg.role),
                        .content = MessageContent{ .text = try allocator.dupe(u8, content_text) },
                        .tool_calls = null,
                        .tool_call_id = null,
                        .msg_id = if (msg.msg_id) |mid| try allocator.dupe(u8, mid) else null,
                        .agent_mode = msg.agent_mode,
                        .cached_size = null,
                    };
                    try cleaned.append(new_msg);
                }
            }
        } else {
            try cleaned.append(try msg.clone(allocator));
        }
    }

    for (messages.items) |*msg| msg.deinit(allocator);
    messages.deinit();
    messages.* = cleaned;
}

fn trimHistory(allocator: std.mem.Allocator, history: *std.ArrayList(Message)) !void {
    if (history.items.len <= MAX_STORED_MESSAGES) return;

    var cut = history.items.len - MAX_STORED_MESSAGES;
    cut = findTurnBoundary(history.items, cut);
    if (cut >= history.items.len) {
        cut = if (history.items.len > MAX_STORED_MESSAGES) history.items.len - MAX_STORED_MESSAGES else 0;
    }
    if (cut > 0) {
        for (history.items[0..cut]) |*msg| msg.deinit(allocator);
        const remaining = try allocator.dupe(Message, history.items[cut..]);
        history.clearRetainingCapacity();
        try history.appendSlice(remaining);
        allocator.free(remaining);
    }
    try stripOrphanedAssistantToolCalls(allocator, history);

    var total_chars: usize = 0;
    for (history.items) |*msg| total_chars += messageCharSize(msg);
    while (total_chars > MAX_MODEL_CONTENT_CHARS * 2 and history.items.len > 1) {
        var cut2 = findTurnBoundary(history.items, 1);
        if (cut2 <= 0 or cut2 >= history.items.len) cut2 = 1;
        for (history.items[0..cut2]) |*msg| msg.deinit(allocator);
        const remaining2 = try allocator.dupe(Message, history.items[cut2..]);
        history.clearRetainingCapacity();
        try history.appendSlice(remaining2);
        allocator.free(remaining2);
        total_chars = 0;
        for (history.items) |*msg| total_chars += messageCharSize(msg);
    }
}

fn buildApiMessages(allocator: std.mem.Allocator, history: []const Message) !std.ArrayList(Message) {
    var result = std.ArrayList(Message).init(allocator);
    for (history) |msg| {
        try result.append(try msg.clone(allocator));
    }

    try stripOrphanedAssistantToolCalls(allocator, &result);

    if (result.items.len > MAX_HISTORY_MESSAGES) {
        var cut = result.items.len - MAX_HISTORY_MESSAGES;
        cut = findTurnBoundary(result.items, cut);
        if (cut >= result.items.len) {
            cut = if (result.items.len > MAX_HISTORY_MESSAGES) result.items.len - MAX_HISTORY_MESSAGES else 0;
        }
        if (cut > 0) {
            for (result.items[0..cut]) |*msg| msg.deinit(allocator);
            const remaining = try allocator.dupe(Message, result.items[cut..]);
            result.clearRetainingCapacity();
            try result.appendSlice(remaining);
            allocator.free(remaining);
        }
        if (result.items.len == 0) return result;
        try stripOrphanedAssistantToolCalls(allocator, &result);
    }

    var total_chars: usize = 0;
    for (result.items) |*msg| total_chars += messageCharSize(msg);

    while (result.items.len > 1 and total_chars > MAX_MODEL_CONTENT_CHARS) {
        var cut = findTurnBoundary(result.items, 1);
        if (cut <= 0 or cut >= result.items.len) cut = 1;
        for (result.items[0..cut]) |*msg| msg.deinit(allocator);
        const remaining = try allocator.dupe(Message, result.items[cut..]);
        result.clearRetainingCapacity();
        try result.appendSlice(remaining);
        allocator.free(remaining);
        if (result.items.len == 0) return result;
        try stripOrphanedAssistantToolCalls(allocator, &result);
        total_chars = 0;
        for (result.items) |*msg| total_chars += messageCharSize(msg);
    }

    return result;
}

fn rollbackSessionTurn(sid: []const u8, msg_id: []const u8) void {
    sessions_mutex.lock();
    defer sessions_mutex.unlock();
    if (sessions_map.getPtr(sid)) |sess| {
        var entry_start: ?usize = null;
        for (sess.messages.items, 0..) |msg, i| {
            if (msg.msg_id) |mid| {
                if (std.mem.eql(u8, mid, msg_id)) {
                    entry_start = i;
                    break;
                }
            }
        }
        if (entry_start) |start| {
            for (sess.messages.items[start..]) |*msg| msg.deinit(global_allocator);
            sess.messages.shrinkRetainingCapacity(start);
        }
        sess.updated_at = nowSeconds();
    }
}

fn getSystemPrompt(allocator: std.mem.Allocator) ![]u8 {
    const now_ms = std.time.milliTimestamp();
    const now_sec = @divTrunc(now_ms, 1000);
    const epoch = std.time.epoch.EpochSeconds{ .secs = @intCast(now_sec) };
    const epoch_day = epoch.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    const total_minutes = @divTrunc(@mod(now_ms, 86400_000), 60000);
    const hour = @divTrunc(total_minutes, 60);
    const minute = @mod(total_minutes, 60);

    return std.fmt.allocPrint(allocator,
        "You are Helium, an intelligent AI assistant.\nCurrent date: {d:0>4}-{d:0>2}-{d:0>2} | Current time: {d:0>2}:{d:0>2} UTC\n\nSEARCH RULES:\n- Use exa_search when the question requires current information, recent events, news, prices, people, sports results, or anything that may have changed.\n- Do NOT search for simple greetings, general knowledge, math, or timeless facts.\n- Provide multiple queries in the 'queries' array for comprehensive coverage.\n\nANSWER RULES:\n- If you searched: base your answer EXCLUSIVELY on the search result summaries received. Do NOT use your internal knowledge.\n- If you did not search: answer directly from your knowledge.\n- Synthesize the summaries into a clear, well-structured answer.\n- Cite sources with URLs where relevant.\n- Match the user's language and tone.\n- Be concise for simple questions, thorough for complex ones.",
        .{
            year_day.year,
            month_day.month.numeric(),
            month_day.day_index + 1,
            hour,
            minute,
        },
    );
}

const AGENT_SYSTEM_PROMPT =
    "You are Helium Agent, an autonomous coding assistant with full filesystem and shell access to a live project.\n\nPROJECT STRUCTURE:\n- main.zig: Backend (Zig)\n- index.html: Full frontend (HTML/CSS/JS single file)\n- static/: Static assets (manifest.json, sw.js)\nTOOLS:\n- read_file: Read file contents (use offset/limit for large files)\n- write_file: Create or overwrite files\n- list_directory: List directory contents\n- execute_command: Run any shell command\n- search_files: Grep for patterns across files\n\nWORKFLOW:\n1. Understand the request fully before making changes\n2. Read relevant files to understand current state\n3. Plan and implement changes\n4. Write COMPLETE file contents - never partial, truncated, or abbreviated\n5. Verify changes by reading back or running commands\n\nRULES:\n- NEVER use placeholders, dummy data, mock implementations, TODOs, or abbreviations like '...'\n- NEVER truncate file output - always write the complete file\n- Always read a file before modifying it\n- Match the user's language (Hungarian if they write Hungarian)\n- Be methodical: explain what you're doing, then do it\n- For large files, read in chunks using offset/limit, then write the full modified content";

const HttpResponse = struct {
    status: u16,
    headers: std.StringHashMap([]const u8),
    body: []u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *HttpResponse) void {
        self.allocator.free(self.body);
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
    }
};

fn httpPost(allocator: std.mem.Allocator, url: []const u8, headers: []const [2][]const u8, body: []const u8) !HttpResponse {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(url);

    var extra_headers = std.ArrayList(std.http.Header).init(allocator);
    defer extra_headers.deinit();
    for (headers) |h| {
        try extra_headers.append(.{ .name = h[0], .value = h[1] });
    }

    var response_body = std.ArrayList(u8).init(allocator);
    defer response_body.deinit();

    const result = try client.fetch(.{
        .method = .POST,
        .location = .{ .uri = uri },
        .headers = .{
            .content_type = .{ .override = "application/json" },
        },
        .extra_headers = extra_headers.items,
        .payload = body,
        .response_storage = .{ .dynamic = &response_body },
    });

    const resp_headers = std.StringHashMap([]const u8).init(allocator);
    return HttpResponse{
        .status = result.status.class() == .success,
        .headers = resp_headers,
        .body = try response_body.toOwnedSlice(),
        .allocator = allocator,
    };
}

fn httpGet(allocator: std.mem.Allocator, url: []const u8, headers: []const [2][]const u8) !HttpResponse {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(url);

    var extra_headers = std.ArrayList(std.http.Header).init(allocator);
    defer extra_headers.deinit();
    for (headers) |h| {
        try extra_headers.append(.{ .name = h[0], .value = h[1] });
    }

    var response_body = std.ArrayList(u8).init(allocator);
    defer response_body.deinit();

    const result = try client.fetch(.{
        .method = .GET,
        .location = .{ .uri = uri },
        .extra_headers = extra_headers.items,
        .response_storage = .{ .dynamic = &response_body },
    });
    const resp_headers = std.StringHashMap([]const u8).init(allocator);
    return HttpResponse{
        .status = result.status.class() == .success,
        .headers = resp_headers,
        .body = try response_body.toOwnedSlice(),
        .allocator = allocator,
    };
}

fn callExaSingle(allocator: std.mem.Allocator, query: []const u8, params: std.json.Value, limiter: *RateLimiter) !std.json.Value {
    if (exa_api_key.len == 0) return error.ExaApiKeyNotConfigured;
    if (query.len == 0) return error.EmptyQuery;

    var payload = std.ArrayList(u8).init(allocator);
    defer payload.deinit();
    var pw = payload.writer();

    try pw.writeByte('{');
    try pw.writeAll("\"query\":");
    try writeJsonString(pw, query);
    try std.fmt.format(pw, ",\"numResults\":{d}", .{EXA_NUM_RESULTS});
    try pw.writeAll(",\"contents\":{\"summary\":{\"query\":");

    const summary_query = if (params == .object) blk: {
        if (params.object.get("summaryQuery")) |sq| {
            if (sq == .string) break :blk sq.string;
        }
        break :blk query;
    } else query;
    try writeJsonString(pw, summary_query);
    try pw.writeAll("}}");

    const search_type = if (params == .object) blk: {
        if (params.object.get("type")) |t| {
            if (t == .string) break :blk t.string;
        }
        break :blk "auto";
    } else "auto";
    try pw.writeAll(",\"type\":");
    try writeJsonString(pw, search_type);

    if (params == .object) {
        if (params.object.get("maxAgeHours")) |mah| {
            if (mah == .integer) {
                try std.fmt.format(pw, ",\"maxAgeHours\":{d}", .{mah.integer});
            }
        }
        for ([_][]const u8{ "category", "startPublishedDate", "endPublishedDate" }) |field| {
            if (params.object.get(field)) |val| {
                if (val == .string and val.string.len > 0) {
                    try pw.writeByte(',');
                    try writeJsonString(pw, field);
                    try pw.writeByte(':');
                    try writeJsonString(pw, val.string);
                }
            }
        }
        for ([_][]const u8{ "includeDomains", "excludeDomains" }) |field| {
            if (params.object.get(field)) |val| {
                if (val == .array and val.array.items.len > 0) {
                    try pw.writeByte(',');
                    try writeJsonString(pw, field);
                    try pw.writeByte(':');
                    try pw.writeByte('[');
                    for (val.array.items, 0..) |item, i| {
                        if (i > 0) try pw.writeByte(',');
                        if (item == .string) try writeJsonString(pw, item.string);
                    }
                    try pw.writeByte(']');
                }
            }
        }
        if (params.object.get("userLocation")) |loc_val| {
            if (loc_val == .string) {
                const loc = std.mem.trim(u8, loc_val.string, " \t\n\r");
                if (loc.len == 2) {
                    var upper_loc: [2]u8 = undefined;
                    upper_loc[0] = std.ascii.toUpper(loc[0]);
                    upper_loc[1] = std.ascii.toUpper(loc[1]);
                    try pw.writeAll(",\"userLocation\":");
                    try writeJsonString(pw, &upper_loc);
                }
            }
        }
        if (isDeepSearchType(search_type)) {
            for ([_][]const u8{ "systemPrompt" }) |field| {
                if (params.object.get(field)) |val| {
                    if (val == .string and val.string.len > 0) {
                        try pw.writeByte(',');
                        try writeJsonString(pw, field);
                        try pw.writeByte(':');
                        try writeJsonString(pw, val.string);
                    }
                }
            }
            if (params.object.get("additionalQueries")) |val| {
                if (val == .array and val.array.items.len > 0) {
                    try pw.writeAll(",\"additionalQueries\":[");
                    for (val.array.items, 0..) |item, i| {
                        if (i > 0) try pw.writeByte(',');
                        if (item == .string) try writeJsonString(pw, item.string);
                    }
                    try pw.writeByte(']');
                }
            }
        }
    }
    try pw.writeByte('}');

    const auth_header = try std.fmt.allocPrint(allocator, "{s}", .{exa_api_key});
    defer allocator.free(auth_header);

    var attempt: usize = 0;
    while (attempt < 3) : (attempt += 1) {
        limiter.acquire();
        exa_limiter.acquire();

        var client = std.http.Client{ .allocator = allocator };
        defer client.deinit();

        const uri = try std.Uri.parse("https://api.exa.ai/search");
        var response_body = std.ArrayList(u8).init(allocator);
        defer response_body.deinit();

        const headers = [_]std.http.Header{
            .{ .name = "x-api-key", .value = auth_header },
            .{ .name = "Content-Type", .value = "application/json" },
        };

        const result = client.fetch(.{
            .method = .POST,
            .location = .{ .uri = uri },
            .extra_headers = &headers,
            .payload = payload.items,
            .response_storage = .{ .dynamic = &response_body },
        }) catch |err| {
            if (attempt < 2) {
                std.time.sleep(@intCast((attempt + 1) * 2 * std.time.ns_per_s));
                continue;
            }
            return err;
        };

        if (result.status == .too_many_requests) {
            if (attempt < 2) {
                std.time.sleep(@intCast((attempt + 1) * 3 * std.time.ns_per_s));
                continue;
            }
            return error.ExaRateLimited;
        }

        if (@intFromEnum(result.status) >= 500) {
            if (attempt < 2) {
                std.time.sleep(@intCast((attempt + 1) * 2 * std.time.ns_per_s));
                continue;
            }
            return error.ExaServerError;
        }

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, response_body.items, .{}) catch return error.ExaInvalidResponse;
        return parsed.value;
    }
    return error.ExaFailed;
}

const ExaMultiResult = struct {
    query: []u8,
    results: ?std.json.Value,
    err: ?[]u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *ExaMultiResult) void {
        self.allocator.free(self.query);
        if (self.err) |e| self.allocator.free(e);
    }
};

const ExaSearchTask = struct {
    query: []const u8,
    params: std.json.Value,
    limiter: *RateLimiter,
    result: ?std.json.Value,
    err_msg: ?[]u8,
    allocator: std.mem.Allocator,
    thread: ?std.Thread,

    fn run(self: *ExaSearchTask) void {
        self.result = callExaSingle(self.allocator, self.query, self.params, self.limiter) catch |err| blk: {
            self.err_msg = std.fmt.allocPrint(self.allocator, "{}", .{err}) catch null;
            break :blk null;
        };
    }
};

fn normalizeExaQueries(allocator: std.mem.Allocator, queries: []const std.json.Value) !std.ArrayList([]u8) {
    var result = std.ArrayList([]u8).init(allocator);
    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    for (queries) |q| {
        if (q != .string) continue;
        const stripped = std.mem.trim(u8, q.string, " \t\n\r");
        if (stripped.len == 0) continue;
        const lower = try std.ascii.allocLowerString(allocator, stripped);
        defer allocator.free(lower);
        if (seen.get(lower) != null) continue;
        const lower_key = try allocator.dupe(u8, lower);
        try seen.put(lower_key, {});
        try result.append(try allocator.dupe(u8, stripped));
    }
    return result;
}

fn formatExaResultsForModel(allocator: std.mem.Allocator, results: []const ExaMultiResult) ![]u8 {
    var lines = std.ArrayList(u8).init(allocator);
    defer lines.deinit();
    var writer = lines.writer();
    var global_count: usize = 0;
    var char_count: usize = 0;
    var truncated = false;

    outer: for (results) |entry| {
        if (entry.err != null and entry.results == null) continue;
        const res_val = entry.results orelse continue;
        if (res_val != .object) continue;
        const res_array = res_val.object.get("results") orelse continue;
        if (res_array != .array) continue;

        for (res_array.array.items) |r| {
            if (r != .object) continue;
            const summary_raw = r.object.get("summary");
            var summary: []const u8 = "";
            if (summary_raw) |sr| {
                if (sr == .string) {
                    summary = sr.string;
                } else if (sr == .object) {
                    if (sr.object.get("text")) |t| {
                        if (t == .string) summary = t.string;
                    }
                }
            }
            const trimmed_summary = std.mem.trim(u8, summary, " \t\n\r");
            if (trimmed_summary.len == 0) continue;
            const capped_summary = trimmed_summary[0..@min(trimmed_summary.len, MAX_SUMMARY_CHARS)];
            global_count += 1;
            const line = try std.fmt.allocPrint(allocator, "[{d}] Query: {s} - {s}", .{ global_count, entry.query, capped_summary });
            defer allocator.free(line);

            if (char_count + line.len + 1 > MAX_MODEL_CONTENT_CHARS) {
                try writer.writeAll("[Search results truncated due to size limit]");
                truncated = true;
                break :outer;
            }
            if (char_count > 0) try writer.writeByte('\n');
            try writer.writeAll(line);
            char_count += line.len + 1;
        }
    }
    const result = try lines.toOwnedSlice();
    if (result.len > MAX_MESSAGE_LENGTH) {
        const truncated_result = try allocator.dupe(u8, result[0..MAX_MESSAGE_LENGTH]);
        allocator.free(result);
        return truncated_result;
    }
    return result;
}

const SseResultItem = struct {
    title: []u8,
    url: []u8,
    summary: []u8,
    published_date: []u8,
    query: []u8,
    err: ?[]u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *SseResultItem) void {
        self.allocator.free(self.title);
        self.allocator.free(self.url);
        self.allocator.free(self.summary);
        self.allocator.free(self.published_date);
        self.allocator.free(self.query);
        if (self.err) |e| self.allocator.free(e);
    }
};

fn formatExaResultsForSse(allocator: std.mem.Allocator, results: []const ExaMultiResult) !struct { items: std.ArrayList(SseResultItem), total: usize } {
    var items = std.ArrayList(SseResultItem).init(allocator);
    var total: usize = 0;

    for (results) |entry| {
        if (entry.results) |res_val| {
            if (res_val == .object) {
                const res_array = res_val.object.get("results") orelse {
                    if (entry.err) |e| {
                        try items.append(SseResultItem{
                            .title = try allocator.dupe(u8, ""),
                            .url = try allocator.dupe(u8, ""),
                            .summary = try allocator.dupe(u8, ""),
                            .published_date = try allocator.dupe(u8, ""),
                            .query = try allocator.dupe(u8, entry.query),
                            .err = try allocator.dupe(u8, e),
                            .allocator = allocator,
                        });
                    }
                    continue;
                };
                if (res_array == .array) {
                    for (res_array.array.items) |r| {
                        if (r != .object) continue;
                        const url = if (r.object.get("url")) |u| if (u == .string) u.string else "" else "";
                        var summary: []const u8 = "";
                        if (r.object.get("summary")) |sr| {
                            if (sr == .string) summary = sr.string;
                            if (sr == .object) {
                                if (sr.object.get("text")) |t| {
                                    if (t == .string) summary = t.string;
                                }
                            }
                        }
                        const title = if (r.object.get("title")) |t| if (t == .string) t.string else "" else "";
                        const pub_date = if (r.object.get("publishedDate")) |pd| if (pd == .string) pd.string else "" else "";

                        try items.append(SseResultItem{
                            .title = try allocator.dupe(u8, title[0..@min(title.len, 120)]),
                            .url = try allocator.dupe(u8, url),
                            .summary = try allocator.dupe(u8, summary[0..@min(summary.len, 400)]),
                            .published_date = try allocator.dupe(u8, pub_date),
                            .query = try allocator.dupe(u8, entry.query),
                            .err = null,
                            .allocator = allocator,
                        });
                        total += 1;
                    }
                }
            }
        }
        if (entry.err) |e| {
            try items.append(SseResultItem{
                .title = try allocator.dupe(u8, ""),
                .url = try allocator.dupe(u8, ""),
                .summary = try allocator.dupe(u8, ""),
                .published_date = try allocator.dupe(u8, ""),
                .query = try allocator.dupe(u8, entry.query),
                .err = try allocator.dupe(u8, e),
                .allocator = allocator,
            });
        }
    }

    return .{ .items = items, .total = total };
}

const SseWriter = struct {
    conn: *std.net.Server.Connection,
    allocator: std.mem.Allocator,

    fn write(self: *SseWriter, data: []const u8) !void {
        const line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{data});
        defer self.allocator.free(line);
        try self.conn.stream.writeAll(line);
    }

    fn ping(self: *SseWriter) !void {
        try self.conn.stream.writeAll(": ping\n\n");
    }

    fn writeComment(self: *SseWriter) !void {
        try self.conn.stream.writeAll(": ping\n\n");
    }
};

fn writeJsonEvent(allocator: std.mem.Allocator, sse: *SseWriter, fields: anytype) !void {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();
    try writer.writeByte('{');
    const fields_info = @typeInfo(@TypeOf(fields)).@"struct";
    inline for (fields_info.fields, 0..) |field, i| {
        if (i > 0) try writer.writeByte(',');
        try writeJsonString(writer, field.name);
        try writer.writeByte(':');
        const val = @field(fields, field.name);
        const ValType = @TypeOf(val);
        switch (@typeInfo(ValType)) {
            .pointer => |ptr| {
                if (ptr.child == u8) {
                    try writeJsonString(writer, val);
                } else {
                    try writer.writeAll("null");
                }
            },
            .int, .comptime_int => {
                try std.fmt.format(writer, "{d}", .{val});
            },
            .bool => {
                try writer.writeAll(if (val) "true" else "false");
            },
            else => {
                try writer.writeAll("null");
            },
        }
    }
    try writer.writeByte('}');
    try sse.write(buf.items);
}

const OpenAIStreamChunk = struct {
    content: ?[]u8,
    reasoning_content: ?[]u8,
    finish_reason: ?[]u8,
    tool_calls: ?std.ArrayList(OpenAIToolCallDelta),
    allocator: std.mem.Allocator,

    fn deinit(self: *OpenAIStreamChunk) void {
        if (self.content) |c| self.allocator.free(c);
        if (self.reasoning_content) |r| self.allocator.free(r);
        if (self.finish_reason) |f| self.allocator.free(f);
        if (self.tool_calls) |*tcs| {
            for (tcs.items) |*tc| tc.deinit(self.allocator);
            tcs.deinit();
        }
    }
};

const OpenAIToolCallDelta = struct {
    index: ?usize,
    id: ?[]u8,
    name: ?[]u8,
    arguments: ?[]u8,

    fn deinit(self: *OpenAIToolCallDelta, allocator: std.mem.Allocator) void {
        if (self.id) |id| allocator.free(id);
        if (self.name) |n| allocator.free(n);
        if (self.arguments) |a| allocator.free(a);
    }
};

fn parseSseLine(allocator: std.mem.Allocator, line: []const u8) !?OpenAIStreamChunk {
    if (!std.mem.startsWith(u8, line, "data: ")) return null;
    const data = line[6..];
    if (std.mem.eql(u8, data, "[DONE]")) return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{}) catch return null;
    defer parsed.deinit();

    const choices = parsed.value.object.get("choices") orelse return null;
    if (choices != .array or choices.array.items.len == 0) return null;

    const choice = choices.array.items[0];
    if (choice != .object) return null;

    const delta = choice.object.get("delta") orelse return null;
    if (delta != .object) return null;

    var chunk = OpenAIStreamChunk{
        .content = null,
        .reasoning_content = null,
        .finish_reason = null,
        .tool_calls = null,
        .allocator = allocator,
    };

    if (delta.object.get("content")) |c| {
        if (c == .string and c.string.len > 0) {
            chunk.content = try allocator.dupe(u8, c.string);
        }
    }
    if (delta.object.get("reasoning_content")) |r| {
        if (r == .string and r.string.len > 0) {
            chunk.reasoning_content = try allocator.dupe(u8, r.string);
        }
    }

    if (choice.object.get("finish_reason")) |fr| {
        if (fr == .string) {
            chunk.finish_reason = try allocator.dupe(u8, fr.string);
        }
    }

    if (delta.object.get("tool_calls")) |tcs_val| {
        if (tcs_val == .array) {
            chunk.tool_calls = std.ArrayList(OpenAIToolCallDelta).init(allocator);
            for (tcs_val.array.items) |tc_val| {
                if (tc_val != .object) continue;
                var tc_delta = OpenAIToolCallDelta{
                    .index = null,
                    .id = null,
                    .name = null,
                    .arguments = null,
                };
                if (tc_val.object.get("index")) |idx| {
                    if (idx == .integer) tc_delta.index = @intCast(idx.integer);
                }
                if (tc_val.object.get("id")) |id| {
                    if (id == .string and id.string.len > 0) {
                        tc_delta.id = try allocator.dupe(u8, id.string);
                    }
                }
                if (tc_val.object.get("function")) |func| {
                    if (func == .object) {
                        if (func.object.get("name")) |n| {
                            if (n == .string and n.string.len > 0) {
                                tc_delta.name = try allocator.dupe(u8, n.string);
                            }
                        }
                        if (func.object.get("arguments")) |a| {
                            if (a == .string and a.string.len > 0) {
                                tc_delta.arguments = try allocator.dupe(u8, a.string);
                            }
                        }
                    }
                }
                try chunk.tool_calls.?.append(tc_delta);
            }
        }
    }

    return chunk;
}

const ToolCallAccumulator = struct {
    id: []u8,
    name: []u8,
    arg_parts: std.ArrayList([]u8),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !*ToolCallAccumulator {
        const self = try allocator.create(ToolCallAccumulator);
        self.* = .{
            .id = try allocator.dupe(u8, ""),
            .name = try allocator.dupe(u8, ""),
            .arg_parts = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
        };
        return self;
    }

    fn deinit(self: *ToolCallAccumulator) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        for (self.arg_parts.items) |part| self.allocator.free(part);
        self.arg_parts.deinit();
        self.allocator.destroy(self);
    }

    fn getArguments(self: *ToolCallAccumulator) ![]u8 {
        var total: usize = 0;
        for (self.arg_parts.items) |part| total += part.len;
        const result = try self.allocator.alloc(u8, total);
        var offset: usize = 0;
        for (self.arg_parts.items) |part| {
            @memcpy(result[offset .. offset + part.len], part);
            offset += part.len;
        }
        return result;
    }
};

fn buildOpenAIRequest(allocator: std.mem.Allocator, model: []const u8, messages: []const Message, max_tokens: usize, include_tools: bool, extra_body_thinking: bool) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    try writer.writeByte('{');
    try writer.writeAll("\"model\":");
    try writeJsonString(writer, model);
    try std.fmt.format(writer, ",\"max_tokens\":{d}", .{max_tokens});
    try writer.writeAll(",\"temperature\":0.1,\"top_p\":0.95");

    if (extra_body_thinking) {
        try writer.writeAll(",\"parse_reasoning\":true,\"chat_template_kwargs\":{\"enable_thinking\":true},\"stream_options\":{\"include_usage\":true}");
    }

    try writer.writeAll(",\"stream\":true");

    if (include_tools) {
        try writer.writeAll(",\"tool_choice\":\"auto\",\"tools\":[");
        try writeExaTool(writer);
        try writer.writeByte(']');
    }

    try writer.writeAll(",\"messages\":[");
    for (messages, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try serializeMessage(writer, msg);
    }
    try writer.writeByte(']');
    try writer.writeByte('}');

    return buf.toOwnedSlice();
}

fn buildAgentOpenAIRequest(allocator: std.mem.Allocator, model: []const u8, messages: []const Message, max_tokens: usize) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    try writer.writeByte('{');
    try writer.writeAll("\"model\":");
    try writeJsonString(writer, model);
    try std.fmt.format(writer, ",\"max_tokens\":{d}", .{max_tokens});
    try writer.writeAll(",\"temperature\":0.1,\"top_p\":0.95");
    try writer.writeAll(",\"stream\":true");
    try writer.writeAll(",\"tool_choice\":\"auto\",\"tools\":[");
    try writeAgentTools(writer);
    try writer.writeByte(']');
    try writer.writeAll(",\"messages\":[");
    for (messages, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try serializeMessage(writer, msg);
    }
    try writer.writeByte(']');
    try writer.writeByte('}');

    return buf.toOwnedSlice();
}

fn writeExaTool(writer: anytype) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"exa_search\",\"description\":\"Use the Exa search engine for one or more web searches to retrieve current, real-time information from the internet.\",\"parameters\":{\"type\":\"object\",\"required\":[\"queries\"],\"properties\":{\"queries\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"description\":\"Array containing one or more search queries.\"},\"type\":{\"type\":\"string\",\"enum\":[\"neural\",\"fast\",\"auto\",\"deep\",\"deep-reasoning\",\"instant\"],\"description\":\"Search type.\"},\"category\":{\"type\":\"string\",\"enum\":[\"company\",\"research paper\",\"news\",\"personal site\",\"financial report\",\"people\"],\"description\":\"Optional category filter.\"},\"startPublishedDate\":{\"type\":\"string\"},\"endPublishedDate\":{\"type\":\"string\"},\"includeDomains\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"excludeDomains\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"userLocation\":{\"type\":\"string\"},\"summaryQuery\":{\"type\":\"string\"},\"maxAgeHours\":{\"type\":\"integer\"},\"additionalQueries\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"systemPrompt\":{\"type\":\"string\"}}}}}");
}

fn writeAgentTools(writer: anytype) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"description\":\"Read the contents of a file. Use offset and limit for large files.\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"File path relative to project root.\"},\"offset\":{\"type\":\"integer\",\"description\":\"Line number to start from (1-indexed).\"},\"limit\":{\"type\":\"integer\",\"description\":\"Max lines to read.\"}}}}}");
    try writer.writeByte(',');
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"description\":\"Write content to a file. Creates parent directories if needed. Overwrites existing file.\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\",\"content\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"File path relative to project root.\"},\"content\":{\"type\":\"string\",\"description\":\"Full file content to write.\"}}}}}");
    try writer.writeByte(',');
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"list_directory\",\"description\":\"List files and directories at a given path.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Directory path relative to project root. Default: root.\"}}}}}");
    try writer.writeByte(',');
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"execute_command\",\"description\":\"Execute a shell command in the project root and return stdout/stderr.\",\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"Shell command to run.\"},\"timeout\":{\"type\":\"integer\",\"description\":\"Timeout in seconds (max 120). Default: 30.\"}}}}}");
    try writer.writeByte(',');
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"search_files\",\"description\":\"Search for a regex pattern in project files using grep.\",\"parameters\":{\"type\":\"object\",\"required\":[\"pattern\"],\"properties\":{\"pattern\":{\"type\":\"string\",\"description\":\"Regex pattern to search for.\"},\"path\":{\"type\":\"string\",\"description\":\"Directory to search in. Default: project root.\"},\"file_glob\":{\"type\":\"string\",\"description\":\"File glob filter, e.g. '*.py' or '*.html'.\"}}}}}");
}

fn callOpenAIStreaming(allocator: std.mem.Allocator, base_url: []const u8, api_key: []const u8, body: []const u8, callback: anytype, ctx: anytype) !void {
    const url_str = try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{base_url});
    defer allocator.free(url_str);

    const auth_header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{api_key});
    defer allocator.free(auth_header);

    var child = std.process.Child.init(&[_][]const u8{
        "curl",
        "-sS",
        "-N",
        "--no-buffer",
        "--fail-with-body",
        "--max-time", "600",
        "-X", "POST",
        url_str,
        "-H", auth_header,
        "-H", "Content-Type: application/json",
        "-H", "Accept: text/event-stream",
        "--data-binary", "@-",
    }, allocator);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    if (child.stdin) |stdin| {
        stdin.writeAll(body) catch {};
        stdin.close();
        child.stdin = null;
    }

    var stderr_buf = std.ArrayList(u8).init(allocator);
    defer stderr_buf.deinit();

    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();

    var read_err: ?anyerror = null;
    if (child.stdout) |stdout| {
        var reader = stdout.reader();
        while (true) {
            const byte = reader.readByte() catch |err| {
                if (err != error.EndOfStream) read_err = err;
                break;
            };
            if (byte == '\n') {
                const line = std.mem.trimRight(u8, line_buf.items, "\r");
                if (line.len > 0) {
                    callback(allocator, line, ctx) catch |err| {
                        read_err = err;
                        break;
                    };
                }
                line_buf.clearRetainingCapacity();
            } else {
                try line_buf.append(byte);
            }
        }
    }

    if (child.stderr) |stderr| {
        stderr.reader().readAllArrayList(&stderr_buf, 65536) catch {};
    }

    const term = child.wait() catch |err| {
        std.log.err("curl wait failed: {}", .{err});
        return err;
    };

    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                std.log.err("curl exit code {d}: {s}", .{ code, stderr_buf.items });
                return error.CurlFailed;
            }
        },
        else => {
            std.log.err("curl terminated abnormally: {}", .{term});
            return error.CurlFailed;
        },
    }

    if (read_err) |err| return err;
}

const ChatStreamContext = struct {
    sse: *SseWriter,
    tool_call_chunks: *std.AutoHashMap(usize, *ToolCallAccumulator),
    finish_reason: *?[]u8,
    turn_content: *std.ArrayList(u8),
    reasoning_started: *bool,
    reasoning_done: *bool,
    allocator: std.mem.Allocator,
};

fn chatStreamCallback(allocator: std.mem.Allocator, line: []const u8, ctx: *ChatStreamContext) !void {
    const chunk = parseSseLine(allocator, line) catch return;
    var c = chunk orelse return;
    defer c.deinit();

    if (c.finish_reason) |fr| {
        if (ctx.finish_reason.*) |old| allocator.free(old);
        ctx.finish_reason.* = try allocator.dupe(u8, fr);
    }

    if (c.reasoning_content) |r| {
        if (!ctx.reasoning_started.*) {
            ctx.reasoning_started.* = true;
            var ev_buf = std.ArrayList(u8).init(allocator);
            defer ev_buf.deinit();
            try ev_buf.writer().writeAll("{\"type\":\"reasoning_start\"}");
            try ctx.sse.write(ev_buf.items);
        }
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"reasoning\",\"text\":");
        try writeJsonString(w, r);
        try w.writeByte('}');
        try ctx.sse.write(ev_buf.items);
    }

    if (c.content) |content| {
        if (ctx.reasoning_started.* and !ctx.reasoning_done.*) {
            ctx.reasoning_done.* = true;
            var ev_buf = std.ArrayList(u8).init(allocator);
            defer ev_buf.deinit();
            try ev_buf.writer().writeAll("{\"type\":\"reasoning_end\"}");
            try ctx.sse.write(ev_buf.items);
        }
        try ctx.turn_content.appendSlice(content);
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"content\",\"text\":");
        try writeJsonString(w, content);
        try w.writeByte('}');
        try ctx.sse.write(ev_buf.items);
    }

    if (c.tool_calls) |tcs| {
        for (tcs.items) |tc| {
            const idx = tc.index orelse continue;
            if (ctx.tool_call_chunks.get(idx) == null) {
                const acc = try ToolCallAccumulator.init(allocator);
                try ctx.tool_call_chunks.put(idx, acc);
            }
            const acc = ctx.tool_call_chunks.get(idx).?;
            if (tc.id) |id| {
                allocator.free(acc.id);
                acc.id = try allocator.dupe(u8, id);
            }
            if (tc.name) |n| {
                allocator.free(acc.name);
                acc.name = try allocator.dupe(u8, n);
            }
            if (tc.arguments) |a| {
                try acc.arg_parts.append(try allocator.dupe(u8, a));
            }
        }
    }
}

fn executeExaToolCall(allocator: std.mem.Allocator, tc_id: []const u8, args_str: []const u8, sid: []const u8) !ExaToolResult {
    var events = std.ArrayList([]u8).init(allocator);

    const parsed_args = std.json.parseFromSlice(std.json.Value, allocator, args_str, .{}) catch {
        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_error\",\"query\":\"\",\"id\":\"{s}\",\"error\":\"Invalid arguments JSON\"}}", .{tc_id});
        try events.append(ev);
        const tool_content = try std.fmt.allocPrint(allocator, "Search failed: invalid arguments JSON", .{});
        const tool_msg = Message{
            .role = try allocator.dupe(u8, "tool"),
            .content = MessageContent{ .text = tool_content },
            .tool_calls = null,
            .tool_call_id = try allocator.dupe(u8, tc_id),
            .msg_id = null,
            .agent_mode = false,
            .cached_size = null,
        };
        return .{ .events = events, .tool_msg = tool_msg, .tool_type = "search" };
    };
    defer parsed_args.deinit();

    if (parsed_args.value != .object) {
        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_error\",\"query\":\"\",\"id\":\"{s}\",\"error\":\"Arguments must be a JSON object\"}}", .{tc_id});
        try events.append(ev);
        const tool_content = try std.fmt.allocPrint(allocator, "Search failed: arguments must be a JSON object", .{});
        const tool_msg = Message{
            .role = try allocator.dupe(u8, "tool"),
            .content = MessageContent{ .text = tool_content },
            .tool_calls = null,
            .tool_call_id = try allocator.dupe(u8, tc_id),
            .msg_id = null,
            .agent_mode = false,
            .cached_size = null,
        };
        return .{ .events = events, .tool_msg = tool_msg, .tool_type = "search" };
    }

    const raw_queries_val = parsed_args.value.object.get("queries") orelse std.json.Value{ .array = std.json.Array.init(allocator) };
    const raw_queries: []const std.json.Value = if (raw_queries_val == .array) raw_queries_val.array.items else &[_]std.json.Value{};

    var normalized_queries = try normalizeExaQueries(allocator, raw_queries);
    defer {
        for (normalized_queries.items) |q| allocator.free(q);
        normalized_queries.deinit();
    }

    if (normalized_queries.items.len == 0) {
        if (parsed_args.value.object.get("query")) |legacy_q| {
            if (legacy_q == .string and legacy_q.string.len > 0) {
                const trimmed = std.mem.trim(u8, legacy_q.string, " \t\n\r");
                if (trimmed.len > 0) {
                    const legacy_vals = [_]std.json.Value{.{ .string = trimmed }};
                    var lq = try normalizeExaQueries(allocator, &legacy_vals);
                    defer {
                        for (lq.items) |q| allocator.free(q);
                        lq.deinit();
                    }
                    for (lq.items) |q| {
                        try normalized_queries.append(try allocator.dupe(u8, q));
                    }
                }
            }
        }
    }

    if (normalized_queries.items.len == 0) {
        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_error\",\"query\":\"\",\"id\":\"{s}\",\"error\":\"No queries provided\"}}", .{tc_id});
        try events.append(ev);
        const tool_content = try std.fmt.allocPrint(allocator, "Search failed: no queries provided", .{});
        const tool_msg = Message{
            .role = try allocator.dupe(u8, "tool"),
            .content = MessageContent{ .text = tool_content },
            .tool_calls = null,
            .tool_call_id = try allocator.dupe(u8, tc_id),
            .msg_id = null,
            .agent_mode = false,
            .cached_size = null,
        };
        return .{ .events = events, .tool_msg = tool_msg, .tool_type = "search" };
    }

    var combined_label = std.ArrayList(u8).init(allocator);
    defer combined_label.deinit();
    const max_label = @min(normalized_queries.items.len, 5);
    for (normalized_queries.items[0..max_label], 0..) |q, i| {
        if (i > 0) try combined_label.appendSlice(" | ");
        try combined_label.appendSlice(q);
    }

    var queries_json = std.ArrayList(u8).init(allocator);
    defer queries_json.deinit();
    try queries_json.append('[');
    for (normalized_queries.items, 0..) |q, i| {
        if (i > 0) try queries_json.append(',');
        try writeJsonString(queries_json.writer(), q);
    }
    try queries_json.append(']');

    const start_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_start\",\"query\":\"{s}\",\"id\":\"{s}\",\"queries\":{s}}}", .{ combined_label.items, tc_id, queries_json.items });
    try events.append(start_ev);

    const limiter = getOrCreateExaLimiter(sid) catch &exa_limiter;

    var results = std.ArrayList(ExaMultiResult).init(allocator);
    defer {
        for (results.items) |*r| r.deinit();
        results.deinit();
    }

    for (normalized_queries.items) |q| {
        if (callExaSingle(allocator, q, parsed_args.value, limiter)) |res| {
            try results.append(ExaMultiResult{
                .query = try allocator.dupe(u8, q),
                .results = res,
                .err = null,
                .allocator = allocator,
            });
        } else |err| {
            const err_msg = try std.fmt.allocPrint(allocator, "{}", .{err});
            try results.append(ExaMultiResult{
                .query = try allocator.dupe(u8, q),
                .results = null,
                .err = err_msg,
                .allocator = allocator,
            });
        }
    }

    const all_errors = blk: {
        if (results.items.len == 0) break :blk false;
        for (results.items) |r| {
            if (r.err == null) break :blk false;
        }
        break :blk true;
    };

    if (all_errors) {
        var errors_buf = std.ArrayList(u8).init(allocator);
        defer errors_buf.deinit();
        for (results.items, 0..) |r, i| {
            if (i > 0) try errors_buf.appendSlice("; ");
            if (r.err) |e| try errors_buf.appendSlice(e);
        }
        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_error\",\"query\":\"{s}\",\"id\":\"{s}\",\"error\":\"{s}\"}}", .{ combined_label.items, tc_id, errors_buf.items });
        try events.append(ev);
    } else {
        var sse_data = try formatExaResultsForSse(allocator, results.items);
        defer {
            for (sse_data.items.items) |*item| item.deinit();
            sse_data.items.deinit();
        }

        var results_json = std.ArrayList(u8).init(allocator);
        defer results_json.deinit();
        const rj = results_json.writer();
        try rj.writeByte('[');
        for (sse_data.items.items, 0..) |item, i| {
            if (i > 0) try rj.writeByte(',');
            try rj.writeByte('{');
            try rj.writeAll("\"title\":");
            try writeJsonString(rj, item.title);
            try rj.writeAll(",\"url\":");
            try writeJsonString(rj, item.url);
            try rj.writeAll(",\"summary\":");
            try writeJsonString(rj, item.summary);
            try rj.writeAll(",\"published_date\":");
            try writeJsonString(rj, item.published_date);
            try rj.writeAll(",\"query\":");
            try writeJsonString(rj, item.query);
            if (item.err) |e| {
                try rj.writeAll(",\"error\":");
                try writeJsonString(rj, e);
            }
            try rj.writeByte('}');
        }
        try rj.writeByte(']');

        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_results\",\"query\":\"{s}\",\"id\":\"{s}\",\"results\":{s},\"total\":{d},\"queries\":{s}}}", .{ combined_label.items, tc_id, results_json.items, sse_data.total, queries_json.items });
        try events.append(ev);
    }

    const model_content = try formatExaResultsForModel(allocator, results.items);

    const tool_msg = Message{
        .role = try allocator.dupe(u8, "tool"),
        .content = MessageContent{ .text = model_content },
        .tool_calls = null,
        .tool_call_id = try allocator.dupe(u8, tc_id),
        .msg_id = null,
        .agent_mode = false,
        .cached_size = null,
    };

    return .{ .events = events, .tool_msg = tool_msg, .tool_type = "search" };
}

fn executeAgentTool(allocator: std.mem.Allocator, name: []const u8, args_str: []const u8) !AgentToolResult {
    var events = std.ArrayList([]u8).init(allocator);

    const parsed_args = std.json.parseFromSlice(std.json.Value, allocator, args_str, .{}) catch {
        const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid JSON arguments\"}}", .{});
        return .{ .result = result, .events = events };
    };
    defer parsed_args.deinit();

    if (parsed_args.value != .object) {
        const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Arguments must be a JSON object\"}}", .{});
        return .{ .result = result, .events = events };
    }

    const args = parsed_args.value.object;

    if (std.mem.eql(u8, name, "read_file")) {
        const rel_path_val = args.get("path") orelse {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid path\"}}", .{});
            return .{ .result = result, .events = events };
        };
        const rel_path = if (rel_path_val == .string) rel_path_val.string else {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid path\"}}", .{});
            return .{ .result = result, .events = events };
        };

        var offset: usize = 1;
        if (args.get("offset")) |o| {
            if (o == .integer and o.integer >= 1) offset = @intCast(o.integer);
        }
        var limit: ?usize = null;
        if (args.get("limit")) |l| {
            if (l == .integer and l.integer > 0) limit = @intCast(l.integer);
        }

        const full_path = try std.fs.path.resolve(allocator, &[_][]const u8{ project_root, rel_path });
        defer allocator.free(full_path);

        if (!pathInProject(full_path)) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Access denied: path outside project\"}}", .{});
            return .{ .result = result, .events = events };
        }

        const file = std.fs.openFileAbsolute(full_path, .{}) catch {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"File not found: {s}\"}}", .{rel_path});
            return .{ .result = result, .events = events };
        };
        defer file.close();

        var file_content = std.ArrayList(u8).init(allocator);
        defer file_content.deinit();

        var line_num: usize = 0;
        var selected_count: usize = 0;
        var char_count: usize = 0;
        var truncated = false;
        var total_lines: usize = 0;

        var buf_reader = std.io.bufferedReader(file.reader());
        var reader = buf_reader.reader();
        var line_buf = std.ArrayList(u8).init(allocator);
        defer line_buf.deinit();

        while (true) {
            reader.streamUntilDelimiter(line_buf.writer(), '\n', null) catch |err| {
                if (err == error.EndOfStream) {
                    if (line_buf.items.len > 0) {
                        line_num += 1;
                        total_lines += 1;
                        if (line_num >= offset) {
                            if (limit == null or selected_count < limit.?) {
                                const remaining = 200000 - char_count;
                                if (remaining > 0) {
                                    const to_append = @min(line_buf.items.len, remaining);
                                    try file_content.appendSlice(line_buf.items[0..to_append]);
                                    char_count += to_append;
                                    if (to_append < line_buf.items.len) truncated = true;
                                    selected_count += 1;
                                } else {
                                    truncated = true;
                                }
                            }
                        }
                    }
                    break;
                }
                break;
            };
            line_num += 1;
            total_lines += 1;
            if (line_num >= offset) {
                if (limit == null or selected_count < limit.?) {
                    const remaining = 200000 - char_count;
                    if (remaining > 0) {
                        const to_append = @min(line_buf.items.len, remaining);
                        try file_content.appendSlice(line_buf.items[0..to_append]);
                        if (to_append < line_buf.items.len) truncated = true;
                        try file_content.append('\n');
                        char_count += to_append + 1;
                        selected_count += 1;
                    } else {
                        truncated = true;
                    }
                } else {
                    total_lines += 0;
                }
            }
            line_buf.clearRetainingCapacity();
        }

        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"file_content\",\"path\":\"{s}\",\"lines\":{d},\"total_lines\":{d},\"truncated\":{}}}", .{ rel_path, selected_count, total_lines, truncated });
        try events.append(ev);

        var result_buf = std.ArrayList(u8).init(allocator);
        defer result_buf.deinit();
        const rw = result_buf.writer();
        try rw.writeByte('{');
        try rw.writeAll("\"content\":");
        try writeJsonString(rw, file_content.items);
        try std.fmt.format(rw, ",\"total_lines\":{d},\"lines_shown\":{d},\"from_line\":{d},\"truncated\":{}", .{ total_lines, selected_count, offset, truncated });
        try rw.writeByte('}');
        return .{ .result = try result_buf.toOwnedSlice(), .events = events };
    } else if (std.mem.eql(u8, name, "write_file")) {
        const rel_path_val = args.get("path") orelse {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid path\"}}", .{});
            return .{ .result = result, .events = events };
        };
        const rel_path = if (rel_path_val == .string) rel_path_val.string else {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid path\"}}", .{});
            return .{ .result = result, .events = events };
        };
        const content_val = args.get("content") orelse std.json.Value{ .string = "" };
        const content = if (content_val == .string) content_val.string else "";

        const full_path = try std.fs.path.resolve(allocator, &[_][]const u8{ project_root, rel_path });
        defer allocator.free(full_path);

        if (!pathInProject(full_path)) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Access denied: path outside project\"}}", .{});
            return .{ .result = result, .events = events };
        }

        if (std.fs.path.dirname(full_path)) |parent| {
            std.fs.makeDirAbsolute(parent) catch {};
        }

        const file = std.fs.createFileAbsolute(full_path, .{}) catch |err| {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{}\"}}", .{err});
            return .{ .result = result, .events = events };
        };
        defer file.close();
        file.writeAll(content) catch |err| {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{}\"}}", .{err});
            return .{ .result = result, .events = events };
        };

        var line_count: usize = 0;
        for (content) |c| {
            if (c == '\n') line_count += 1;
        }
        if (content.len > 0 and content[content.len - 1] != '\n') line_count += 1;

        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"file_written\",\"path\":\"{s}\",\"size\":{d},\"lines\":{d}}}", .{ rel_path, content.len, line_count });
        try events.append(ev);

        const result = try std.fmt.allocPrint(allocator, "{{\"success\":true,\"path\":\"{s}\",\"bytes\":{d},\"lines\":{d}}}", .{ rel_path, content.len, line_count });
        return .{ .result = result, .events = events };
    } else if (std.mem.eql(u8, name, "list_directory")) {
        const rel_path_val = args.get("path") orelse std.json.Value{ .string = "." };
        const rel_path = if (rel_path_val == .string) rel_path_val.string else ".";

        const full_path = try std.fs.path.resolve(allocator, &[_][]const u8{ project_root, rel_path });
        defer allocator.free(full_path);

        if (!pathInProject(full_path)) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Access denied: path outside project\"}}", .{});
            return .{ .result = result, .events = events };
        }

        var dir = std.fs.openDirAbsolute(full_path, .{ .iterate = true }) catch {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Not a directory: {s}\"}}", .{rel_path});
            return .{ .result = result, .events = events };
        };
        defer dir.close();

        var entries_buf = std.ArrayList(u8).init(allocator);
        defer entries_buf.deinit();
        try entries_buf.append('[');
        var count: usize = 0;

        var names = std.ArrayList([]u8).init(allocator);
        defer {
            for (names.items) |n| allocator.free(n);
            names.deinit();
        }

        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.name, ".") and
                !std.mem.eql(u8, entry.name, ".replit") and
                !std.mem.eql(u8, entry.name, ".env")) continue;
            try names.append(try allocator.dupe(u8, entry.name));
        }

        std.sort.block([]u8, names.items, {}, struct {
            fn lessThan(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        for (names.items) |entry_name| {
            const entry_path = try std.fs.path.join(allocator, &[_][]const u8{ full_path, entry_name });
            defer allocator.free(entry_path);

            if (count > 0) try entries_buf.append(',');
            try entries_buf.append('{');
            try entries_buf.writer().writeAll("\"name\":");
            try writeJsonString(entries_buf.writer(), entry_name);

            const stat = std.fs.cwd().statFile(entry_path) catch null;
            if (stat) |s| {
                if (s.kind == .directory) {
                    try entries_buf.writer().writeAll(",\"type\":\"dir\"");
                } else {
                    try std.fmt.format(entries_buf.writer(), ",\"type\":\"file\",\"size\":{d}", .{s.size});
                }
            } else {
                try entries_buf.writer().writeAll(",\"type\":\"unknown\"");
            }
            try entries_buf.append('}');
            count += 1;
        }
        try entries_buf.append(']');

        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"directory_listing\",\"path\":\"{s}\",\"count\":{d}}}", .{ rel_path, count });
        try events.append(ev);

        const result = try std.fmt.allocPrint(allocator, "{{\"entries\":{s},\"count\":{d}}}", .{ entries_buf.items, count });
        return .{ .result = result, .events = events };
    } else if (std.mem.eql(u8, name, "execute_command")) {
        const command_val = args.get("command") orelse {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid command\"}}", .{});
            return .{ .result = result, .events = events };
        };
        const command = if (command_val == .string) std.mem.trim(u8, command_val.string, " \t\n\r") else "";
        if (command.len == 0) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid command\"}}", .{});
            return .{ .result = result, .events = events };
        }
        var timeout_sec: u32 = 30;
        if (args.get("timeout")) |t| {
            if (t == .integer) timeout_sec = @intCast(@min(@max(t.integer, 1), 120));
        }

        const ev_start = try std.fmt.allocPrint(allocator, "{{\"type\":\"command_start\",\"command\":\"{s}\"}}", .{command});
        try events.append(ev_start);

        const result_proc = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{ "sh", "-c", command },
            .cwd = project_root,
            .max_output_bytes = 200000,
        }) catch |err| {
            const err_msg = try std.fmt.allocPrint(allocator, "{}", .{err});
            defer allocator.free(err_msg);
            const ev_out = try std.fmt.allocPrint(allocator, "{{\"type\":\"command_output\",\"command\":\"{s}\",\"output\":\"{s}\",\"exit_code\":-1}}", .{ command, err_msg });
            try events.append(ev_out);
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\",\"exit_code\":-1}}", .{err_msg});
            return .{ .result = result, .events = events };
        };
        defer allocator.free(result_proc.stdout);
        defer allocator.free(result_proc.stderr);

        const stdout = result_proc.stdout[0..@min(result_proc.stdout.len, 100000)];
        const stderr = result_proc.stderr[0..@min(result_proc.stderr.len, 50000)];

        var combined = std.ArrayList(u8).init(allocator);
        defer combined.deinit();
        try combined.appendSlice(stdout);
        if (stderr.len > 0) {
            if (combined.items.len > 0) try combined.append('\n');
            try combined.appendSlice(stderr);
        }

        var ev_out_buf = std.ArrayList(u8).init(allocator);
        defer ev_out_buf.deinit();
        const ew = ev_out_buf.writer();
        try ew.writeByte('{');
        try ew.writeAll("\"type\":\"command_output\",\"command\":");
        try writeJsonString(ew, command);
        try ew.writeAll(",\"output\":");
        try writeJsonString(ew, combined.items);
        try std.fmt.format(ew, ",\"exit_code\":{d}", .{result_proc.term.Exited});
        try ew.writeByte('}');
        try events.append(try ev_out_buf.toOwnedSlice());

        var result_buf = std.ArrayList(u8).init(allocator);
        defer result_buf.deinit();
        const rw = result_buf.writer();
        try rw.writeByte('{');
        try rw.writeAll("\"stdout\":");
        try writeJsonString(rw, stdout);
        try rw.writeAll(",\"stderr\":");
        try writeJsonString(rw, stderr);
        try std.fmt.format(rw, ",\"exit_code\":{d}", .{result_proc.term.Exited});
        try rw.writeByte('}');
        return .{ .result = try result_buf.toOwnedSlice(), .events = events };
    } else if (std.mem.eql(u8, name, "search_files")) {
        const pattern_val = args.get("pattern") orelse {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid pattern\"}}", .{});
            return .{ .result = result, .events = events };
        };
        const pattern = if (pattern_val == .string) pattern_val.string else "";
        if (pattern.len == 0) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Invalid pattern\"}}", .{});
            return .{ .result = result, .events = events };
        }
        const rel_path_val = args.get("path") orelse std.json.Value{ .string = "." };
        const rel_path = if (rel_path_val == .string) rel_path_val.string else ".";
        const file_glob_val = args.get("file_glob") orelse std.json.Value{ .string = "" };
        const file_glob = if (file_glob_val == .string) file_glob_val.string else "";

        const full_path = try std.fs.path.resolve(allocator, &[_][]const u8{ project_root, rel_path });
        defer allocator.free(full_path);

        if (!pathInProject(full_path)) {
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Access denied: path outside project\"}}", .{});
            return .{ .result = result, .events = events };
        }

        var cmd_parts = std.ArrayList([]const u8).init(allocator);
        defer cmd_parts.deinit();
        try cmd_parts.appendSlice(&[_][]const u8{ "grep", "-rn", "--color=never" });
        if (file_glob.len > 0) {
            try cmd_parts.appendSlice(&[_][]const u8{ "--include", file_glob });
        }
        try cmd_parts.appendSlice(&[_][]const u8{
            "--exclude-dir=.git",
            "--exclude-dir=node_modules",
            "--exclude-dir=__pycache__",
            "--exclude-dir=.local",
            "--exclude-dir=.replit",
            "--",
            pattern,
            full_path,
        });

        const result_proc = std.process.Child.run(.{
            .allocator = allocator,
            .argv = cmd_parts.items,
            .cwd = project_root,
            .max_output_bytes = 200000,
        }) catch |err| {
            const err_msg = try std.fmt.allocPrint(allocator, "{}", .{err});
            defer allocator.free(err_msg);
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{err_msg});
            return .{ .result = result, .events = events };
        };
        defer allocator.free(result_proc.stdout);
        defer allocator.free(result_proc.stderr);

        const exit_code = result_proc.term.Exited;
        if (exit_code != 0 and exit_code != 1) {
            const err_msg = std.mem.trim(u8, result_proc.stderr, " \t\n\r");
            const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\",\"exit_code\":{d}}}", .{ err_msg, exit_code });
            return .{ .result = result, .events = events };
        }

        const output = result_proc.stdout[0..@min(result_proc.stdout.len, 100000)];
        var line_count: usize = 0;
        var it2 = std.mem.splitScalar(u8, output, '\n');
        while (it2.next()) |_| line_count += 1;
        if (line_count > 0 and output.len > 0 and output[output.len - 1] != '\n') {
        } else if (output.len > 0) {
            line_count -= 1;
        }

        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_result\",\"pattern\":\"{s}\",\"matches\":{d}}}", .{ pattern, line_count });
        try events.append(ev);

        var result_buf = std.ArrayList(u8).init(allocator);
        defer result_buf.deinit();
        const rw = result_buf.writer();
        try rw.writeByte('{');
        try rw.writeAll("\"results\":");
        try writeJsonString(rw, output);
        try std.fmt.format(rw, ",\"match_count\":{d}", .{line_count});
        try rw.writeByte('}');
        return .{ .result = try result_buf.toOwnedSlice(), .events = events };
    }

    const result = try std.fmt.allocPrint(allocator, "{{\"error\":\"Unknown tool: {s}\"}}", .{name});
    return .{ .result = result, .events = events };
}

const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *HttpRequest) void {
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        self.allocator.free(self.body);
    }
};

fn parseHttpRequest(allocator: std.mem.Allocator, reader: anytype) !HttpRequest {
    var first_line_buf = std.ArrayList(u8).init(allocator);
    defer first_line_buf.deinit();

    while (true) {
        const byte = try reader.readByte();
        if (byte == '\n') break;
        if (byte != '\r') try first_line_buf.append(byte);
    }

    var parts = std.mem.splitScalar(u8, first_line_buf.items, ' ');
    const method = parts.next() orelse return error.InvalidRequest;
    const path_and_query = parts.next() orelse return error.InvalidRequest;

    const path = if (std.mem.indexOf(u8, path_and_query, "?")) |q_idx|
        path_and_query[0..q_idx]
    else
        path_and_query;

    var headers = std.StringHashMap([]const u8).init(allocator);
    var content_length: usize = 0;

    while (true) {
        var header_line = std.ArrayList(u8).init(allocator);
        defer header_line.deinit();

        while (true) {
            const byte = try reader.readByte();
            if (byte == '\n') break;
            if (byte != '\r') try header_line.append(byte);
        }

        if (header_line.items.len == 0) break;

        if (std.mem.indexOf(u8, header_line.items, ":")) |colon_idx| {
            const header_name = std.mem.trim(u8, header_line.items[0..colon_idx], " \t");
            const header_value = std.mem.trim(u8, header_line.items[colon_idx + 1 ..], " \t");

            var lower_name = try allocator.alloc(u8, header_name.len);
            for (header_name, 0..) |c, i| {
                lower_name[i] = std.ascii.toLower(c);
            }

            if (std.mem.eql(u8, lower_name, "content-length")) {
                content_length = std.fmt.parseInt(usize, header_value, 10) catch 0;
            }

            const owned_value = try allocator.dupe(u8, header_value);
            try headers.put(lower_name, owned_value);
        }
    }

    var body: []u8 = &[_]u8{};
    if (content_length > 0 and content_length <= 100 * 1024 * 1024) {
        body = try allocator.alloc(u8, content_length);
        try reader.readNoEof(body);
    }

    return HttpRequest{
        .method = try allocator.dupe(u8, method),
        .path = try allocator.dupe(u8, path),
        .headers = headers,
        .body = body,
        .allocator = allocator,
    };
}

fn sendHttpResponse(writer: anytype, status: u16, status_text: []const u8, content_type: []const u8, extra_headers: []const [2][]const u8, body: []const u8) !void {
    try std.fmt.format(writer, "HTTP/1.1 {d} {s}\r\n", .{ status, status_text });
    try std.fmt.format(writer, "Content-Type: {s}\r\n", .{content_type});
    try std.fmt.format(writer, "Content-Length: {d}\r\n", .{body.len});
    try writer.writeAll("Connection: close\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n");
    try writer.writeAll("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    for (extra_headers) |h| {
        try std.fmt.format(writer, "{s}: {s}\r\n", .{ h[0], h[1] });
    }
    try writer.writeAll("\r\n");
    try writer.writeAll(body);
}

fn sendSseHeaders(writer: anytype) !void {
    try writer.writeAll("HTTP/1.1 200 OK\r\n");
    try writer.writeAll("Content-Type: text/event-stream\r\n");
    try writer.writeAll("Cache-Control: no-cache\r\n");
    try writer.writeAll("X-Accel-Buffering: no\r\n");
    try writer.writeAll("Connection: keep-alive\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n");
    try writer.writeAll("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    try writer.writeAll("\r\n");
}

fn sendSseEvent(writer: anytype, data: []const u8) !void {
    try std.fmt.format(writer, "data: {s}\n\n", .{data});
}

fn sendSsePing(writer: anytype) !void {
    try writer.writeAll(": ping\n\n");
}

fn urlDecode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '%' and i + 2 < s.len) {
            const hex = s[i + 1 .. i + 3];
            const byte = std.fmt.parseInt(u8, hex, 16) catch {
                try result.append(s[i]);
                i += 1;
                continue;
            };
            try result.append(byte);
            i += 3;
        } else if (s[i] == '+') {
            try result.append(' ');
            i += 1;
        } else {
            try result.append(s[i]);
            i += 1;
        }
    }
    return result.toOwnedSlice();
}

fn urlEncode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    for (s) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try result.append(c);
        } else {
            try std.fmt.format(result.writer(), "%{X:0>2}", .{c});
        }
    }
    return result.toOwnedSlice();
}

fn extractImageKeysFromMessages(allocator: std.mem.Allocator, msgs: []const Message) !std.ArrayList([]u8) {
    var keys = std.ArrayList([]u8).init(allocator);
    for (msgs) |msg| {
        switch (msg.content) {
            .parts => |parts| {
                for (parts.items) |part| {
                    if (std.mem.eql(u8, part.type, "image_ref")) {
                        if (part.key) |k| {
                            try keys.append(try allocator.dupe(u8, k));
                        }
                    }
                }
            },
            else => {},
        }
    }
    return keys;
}

fn extractPreviewAndCount(msgs: []const Message) struct { preview: []const u8, count: usize } {
    var last_user: ?*const Message = null;
    for (msgs) |*msg| {
        if (std.mem.eql(u8, msg.role, "user")) {
            last_user = msg;
        }
    }

    var preview: []const u8 = "";
    if (last_user) |msg| {
        switch (msg.content) {
            .text => |t| preview = t,
            .parts => |parts| {
                for (parts.items) |part| {
                    if (std.mem.eql(u8, part.type, "text")) {
                        if (part.text) |t| {
                            preview = t;
                            break;
                        }
                    }
                }
                if (preview.len == 0) {
                    for (parts.items) |part| {
                        if (std.mem.eql(u8, part.type, "image_url") or
                            std.mem.eql(u8, part.type, "image_ref") or
                            std.mem.eql(u8, part.type, "image_inline"))
                        {
                            preview = "[kep]";
                            break;
                        }
                    }
                }
            },
        }
    }

    var count: usize = 0;
    for (msgs) |msg| {
        if (std.mem.eql(u8, msg.role, "user") or std.mem.eql(u8, msg.role, "assistant")) {
            if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                const has_content = switch (msg.content) {
                    .text => |t| t.len > 0,
                    .parts => |p| p.items.len > 0,
                };
                if (!has_content) continue;
            }
            count += 1;
        }
    }

    const capped = if (preview.len > 80) preview[0..80] else preview;
    return .{ .preview = capped, .count = count };
}

fn msgsDbToReadable(allocator: std.mem.Allocator, msgs: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    try writer.writeByte('[');
    for (msgs, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeByte('{');
        try writer.writeAll("\"role\":");
        try writeJsonString(writer, msg.role);

        if (std.mem.eql(u8, msg.role, "tool")) {
            const content_str = switch (msg.content) {
                .text => |t| t,
                .parts => "",
            };
            try writer.writeAll(",\"content\":");
            try writeJsonString(writer, content_str);
            if (msg.tool_call_id) |id| {
                try writer.writeAll(",\"tool_call_id\":");
                try writeJsonString(writer, id);
            }
            try writer.writeAll(",\"is_tool_result\":true,\"has_image\":false,\"image_keys\":[]");
        } else {
            switch (msg.content) {
                .text => |t| {
                    try writer.writeAll(",\"content\":");
                    try writeJsonString(writer, t);
                    try writer.writeAll(",\"has_image\":false,\"image_keys\":[]");
                },
                .parts => |parts| {
                    var has_image = false;
                    var text_parts = std.ArrayList([]const u8).init(allocator);
                    defer text_parts.deinit();
                    var image_keys = std.ArrayList([]const u8).init(allocator);
                    defer image_keys.deinit();

                    for (parts.items) |part| {
                        const ptype = part.type;
                        if (std.mem.eql(u8, ptype, "image_ref") or
                            std.mem.eql(u8, ptype, "image_url") or
                            std.mem.eql(u8, ptype, "image_inline"))
                        {
                            has_image = true;
                            if (std.mem.eql(u8, ptype, "image_ref")) {
                                if (part.key) |k| try image_keys.append(k);
                            }
                        } else if (std.mem.eql(u8, ptype, "text")) {
                            if (part.text) |t| try text_parts.append(t);
                        }
                    }

                    var combined = std.ArrayList(u8).init(allocator);
                    defer combined.deinit();
                    for (text_parts.items, 0..) |t, ti| {
                        if (ti > 0) try combined.append('\n');
                        try combined.appendSlice(t);
                    }

                    try writer.writeAll(",\"content\":");
                    try writeJsonString(writer, combined.items);
                    try std.fmt.format(writer, ",\"has_image\":{}", .{has_image});
                    try writer.writeAll(",\"image_keys\":[");
                    for (image_keys.items, 0..) |k, ki| {
                        if (ki > 0) try writer.writeByte(',');
                        try writeJsonString(writer, k);
                    }
                    try writer.writeByte(']');

                    try writer.writeAll(",\"parts\":[");
                    for (parts.items, 0..) |part, pi| {
                        if (pi > 0) try writer.writeByte(',');
                        try serializeContentPart(writer, part);
                    }
                    try writer.writeByte(']');
                },
            }

            if (msg.tool_calls) |tcs| {
                try writer.writeAll(",\"tool_calls\":[");
                for (tcs.items, 0..) |tc, tci| {
                    if (tci > 0) try writer.writeByte(',');
                    try writer.writeAll("{\"id\":");
                    try writeJsonString(writer, tc.id);
                    try writer.writeAll(",\"type\":");
                    try writeJsonString(writer, tc.type);
                    try writer.writeAll(",\"function\":{\"name\":");
                    try writeJsonString(writer, tc.function.name);
                    try writer.writeAll(",\"arguments\":");
                    try writeJsonString(writer, tc.function.arguments);
                    try writer.writeAll("}}");
                }
                try writer.writeByte(']');
            }

            if (msg.agent_mode) {
                try writer.writeAll(",\"agentMode\":true");
            }
        }

        try writer.writeByte('}');
    }
    try writer.writeByte(']');

    return buf.toOwnedSlice();
}

const ConnectionContext = struct {
    conn: std.net.Server.Connection,
    allocator: std.mem.Allocator,
};

fn handleConnection(ctx: ConnectionContext) void {
    defer ctx.conn.stream.close();
    handleRequest(ctx.allocator, ctx.conn) catch |err| {
        std.log.err("Request error: {}", .{err});
    };
}

fn handleRequest(allocator: std.mem.Allocator, conn: std.net.Server.Connection) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const reader = conn.stream.reader();

    var first_line_buf = std.ArrayList(u8).init(arena_alloc);
    while (true) {
        const byte = reader.readByte() catch return;
        if (byte == '\n') break;
        if (byte != '\r') try first_line_buf.append(byte);
    }

    var parts_iter = std.mem.splitScalar(u8, first_line_buf.items, ' ');
    const method_str = parts_iter.next() orelse return;
    const path_full = parts_iter.next() orelse return;

    const path_only = if (std.mem.indexOf(u8, path_full, "?")) |qi| path_full[0..qi] else path_full;

    var headers = std.StringHashMap([]const u8).init(arena_alloc);
    var content_length: usize = 0;

    while (true) {
        var header_line = std.ArrayList(u8).init(arena_alloc);
        while (true) {
            const byte = reader.readByte() catch return;
            if (byte == '\n') break;
            if (byte != '\r') try header_line.append(byte);
        }
        if (header_line.items.len == 0) break;
        if (std.mem.indexOf(u8, header_line.items, ":")) |ci| {
            const hn = std.mem.trim(u8, header_line.items[0..ci], " \t");
            const hv = std.mem.trim(u8, header_line.items[ci + 1 ..], " \t");
            var lower_hn = try arena_alloc.alloc(u8, hn.len);
            for (hn, 0..) |c, idx| lower_hn[idx] = std.ascii.toLower(c);
            if (std.mem.eql(u8, lower_hn, "content-length")) {
                content_length = std.fmt.parseInt(usize, hv, 10) catch 0;
            }
            try headers.put(lower_hn, try arena_alloc.dupe(u8, hv));
        }
    }

    var body: []u8 = &[_]u8{};
    if (content_length > 0 and content_length <= 50 * 1024 * 1024) {
        body = try arena_alloc.alloc(u8, content_length);
        try reader.readNoEof(body);
    }

    const method = method_str;

    if (std.mem.eql(u8, method, "OPTIONS")) {
        const resp = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nContent-Length: 0\r\n\r\n";
        conn.stream.writeAll(resp) catch {};
        return;
    }

    const writer = conn.stream.writer();

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/")) {
        const f = std.fs.openFileAbsolute(index_file, .{}) catch {
            const body_text = "{\"error\":\"index.html not found\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 50 * 1024 * 1024) catch {
            const body_text = "{\"error\":\"Failed to read index.html\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        try sendHttpResponse(writer, 200, "OK", "text/html; charset=utf-8", &[_][2][]const u8{
            .{ "Cache-Control", "no-cache, no-store, must-revalidate" },
        }, content);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/sw.js")) {
        const f = std.fs.openFileAbsolute(static_sw_file, .{}) catch {
            const body_text = "{\"detail\":\"Service worker not found\"}";
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 10 * 1024 * 1024) catch {
            const body_text = "{\"error\":\"Failed to read sw.js\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        try sendHttpResponse(writer, 200, "OK", "application/javascript", &[_][2][]const u8{
            .{ "Service-Worker-Allowed", "/" },
        }, content);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/health")) {
        const ai_ok = hpc_ai_api_key.len > 0;
        const agent_ok = fireworks_api_key.len > 0;
        const status_str = if (ai_ok) "ok" else "degraded";
        const resp_body = try std.fmt.allocPrint(arena_alloc, "{{\"status\":\"{s}\",\"obj_store\":{},\"ai\":{},\"agent\":{}}}", .{
            status_str,
            obj_bucket.len > 0,
            ai_ok,
            agent_ok,
        });
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_body);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/static/")) {
        const rel = path_only[8..];
        const file_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ static_dir, rel });
        const f = std.fs.openFileAbsolute(file_path, .{}) catch {
            const body_text = "{\"detail\":\"Not found\"}";
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 50 * 1024 * 1024) catch {
            const body_text = "{\"error\":\"Failed to read file\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        const ext = std.fs.path.extension(rel);
        const ct = if (std.mem.eql(u8, ext, ".js")) "application/javascript" else if (std.mem.eql(u8, ext, ".css")) "text/css" else if (std.mem.eql(u8, ext, ".json")) "application/json" else "application/octet-stream";
        try sendHttpResponse(writer, 200, "OK", ct, &[_][2][]const u8{}, content);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/api/image/")) {
        const key_encoded = path_only[11..];
        const key = try urlDecode(arena_alloc, key_encoded);
        const image_dir = try std.fs.path.join(arena_alloc, &[_][]const u8{ project_root, "images" });
        const image_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ image_dir, key });
        const f = std.fs.openFileAbsolute(image_path, .{}) catch {
            const body_text = "{\"detail\":\"Image not found\"}";
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        defer f.close();
        const data = f.readToEndAlloc(arena_alloc, MAX_IMAGE_BYTES) catch {
            const body_text = "{\"error\":\"Failed to read image\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        const detected = detectImageFormat(data);
        var mime = detected.mime;
        if (std.mem.eql(u8, mime, "application/octet-stream")) {
            const ext = std.fs.path.extension(key);
            if (extToMime(ext)) |m| mime = m;
        }
        try sendHttpResponse(writer, 200, "OK", mime, &[_][2][]const u8{
            .{ "Cache-Control", "public, max-age=86400" },
        }, data);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/api/sessions")) {
        var result_buf = std.ArrayList(u8).init(arena_alloc);
        defer result_buf.deinit();
        const rw = result_buf.writer();

        const now = nowSeconds();
        var deleted_set = std.StringHashMap(void).init(arena_alloc);
        {
            deleted_sessions_mutex.lock();
            defer deleted_sessions_mutex.unlock();
            var it = deleted_sessions_map.iterator();
            while (it.next()) |entry| {
                if (now - entry.value_ptr.* <= SESSION_TTL) {
                    deleted_set.put(entry.key_ptr.*, {}) catch {};
                }
            }
        }

        const SessionEntry = struct { sid: []const u8, updated_at: f64, preview: []const u8, count: usize };
        var entries = std.ArrayList(SessionEntry).init(arena_alloc);

        {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            var it = sessions_map.iterator();
            while (it.next()) |entry| {
                if (deleted_set.get(entry.key_ptr.*) != null) continue;
                const pc = extractPreviewAndCount(entry.value_ptr.*.messages.items);
                entries.append(.{
                    .sid = entry.key_ptr.*,
                    .updated_at = entry.value_ptr.*.updated_at,
                    .preview = pc.preview,
                    .count = pc.count,
                }) catch {};
            }
        }

        std.sort.block(SessionEntry, entries.items, {}, struct {
            fn lessThan(_: void, a: SessionEntry, b: SessionEntry) bool {
                return a.updated_at > b.updated_at;
            }
        }.lessThan);

        const limit = @min(entries.items.len, 100);

        try rw.writeAll("{\"sessions\":[");
        for (entries.items[0..limit], 0..) |entry, i| {
            if (i > 0) try rw.writeByte(',');
            try rw.writeByte('{');
            try rw.writeAll("\"session_id\":");
            try writeJsonString(rw, entry.sid);
            try rw.writeAll(",\"preview\":");
            try writeJsonString(rw, entry.preview);
            try std.fmt.format(rw, ",\"count\":{d}", .{entry.count});
            try rw.writeByte('}');
        }
        try rw.writeAll("]}");

        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, result_buf.items);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/api/session/")) {
        const sid = path_only[13..];
        if (isDeleted(sid)) {
            const body_text = "{\"detail\":\"Session not found\"}";
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
            return;
        }
        var msgs: ?[]Message = null;
        {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.getPtr(sid)) |sess| {
                msgs = try arena_alloc.dupe(Message, sess.messages.items);
            }
        }
        if (msgs == null) {
            const body_text = "{\"detail\":\"Session not found\"}";
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
            return;
        }
        const readable = try msgsDbToReadable(arena_alloc, msgs.?);
        var resp_buf = std.ArrayList(u8).init(arena_alloc);
        try resp_buf.writer().writeAll("{\"session_id\":");
        try writeJsonString(resp_buf.writer(), sid);
        try resp_buf.writer().writeAll(",\"messages\":");
        try resp_buf.writer().writeAll(readable);
        try resp_buf.writer().writeByte('}');
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
        return;
    }

    if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path_only, "/api/session/")) {
        const sid = path_only[13..];
        markDeleted(sid);
        const session_lock = getOrCreateSessionLock(sid) catch {
            const body_text = "{\"detail\":\"Internal error\"}";
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, body_text);
            return;
        };
        session_lock.lock();
        defer session_lock.unlock();
        {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.fetchRemove(sid)) |kv| {
                var sess = kv.value;
                sess.deinit(global_allocator);
                global_allocator.free(kv.key);
                saveSessionsToDisk();
            } else {
                unmarkDeleted(sid);
                {
                    session_locks_guard.lock();
                    defer session_locks_guard.unlock();
                    if (session_locks_map.fetchRemove(sid)) |kv2| {
                        global_allocator.free(kv2.key);
                    }
                }
                const body_text = "{\"detail\":\"Session not found\"}";
                try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
                return;
            }
        }
        {
            session_locks_guard.lock();
            defer session_locks_guard.unlock();
            if (session_locks_map.fetchRemove(sid)) |kv2| {
                global_allocator.free(kv2.key);
            }
        }
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, "{\"cleared\":true}");
        return;
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/chat")) {
        try handleChatEndpoint(arena_alloc, conn, body);
        return;
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/agent")) {
        try handleAgentEndpoint(arena_alloc, conn, body);
        return;
    }

    const body_text = "{\"detail\":\"Not found\"}";
    try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, body_text);
}

const ChatRequestData = struct {
    session_id: ?[]u8,
    message: []u8,
    images: ?std.json.Array,
    agent_mode: bool,

    fn deinit(self: *ChatRequestData, allocator: std.mem.Allocator) void {
        if (self.session_id) |s| allocator.free(s);
        allocator.free(self.message);
    }
};

fn parseChatRequest(allocator: std.mem.Allocator, body: []const u8) !ChatRequestData {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidRequest;
    const obj = parsed.value.object;

    const message_val = obj.get("message") orelse return error.MissingMessage;
    const message_src = if (message_val == .string) message_val.string else return error.MissingMessage;

    const trimmed = std.mem.trim(u8, message_src, " \t\n\r");
    if (trimmed.len == 0) return error.EmptyMessage;
    if (message_src.len > MAX_MESSAGE_LENGTH) return error.MessageTooLong;

    const message = try allocator.dupe(u8, message_src);
    errdefer allocator.free(message);

    const session_id: ?[]u8 = if (obj.get("session_id")) |sv| blk: {
        if (sv == .string and sv.string.len > 0) {
            break :blk try allocator.dupe(u8, sv.string);
        }
        break :blk null;
    } else null;

    const agent_mode = if (obj.get("agentMode")) |am|
        if (am == .bool) am.bool else false
    else
        false;

    return ChatRequestData{
        .session_id = session_id,
        .message = message,
        .images = null,
        .agent_mode = agent_mode,
    };
}

fn getOrCreateSession(allocator: std.mem.Allocator, session_id: ?[]const u8) !struct { sid: []u8, history: std.ArrayList(Message) } {
    var sid: []u8 = undefined;
    if (session_id) |given_sid| {
        if (isDeleted(given_sid)) {
            sid = try generateUuid(allocator);
        } else {
            sid = try allocator.dupe(u8, given_sid);
        }
    } else {
        sid = try generateUuid(allocator);
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (sessions_map.getPtr(sid)) |sess| {
            sess.updated_at = nowSeconds();
            var history = std.ArrayList(Message).init(allocator);
            for (sess.messages.items) |msg| {
                try history.append(try msg.clone(allocator));
            }
            return .{ .sid = sid, .history = history };
        }
    }

    evictSessions();

    sessions_mutex.lock();
    defer sessions_mutex.unlock();

    if (sessions_map.getPtr(sid)) |sess| {
        sess.updated_at = nowSeconds();
        var history = std.ArrayList(Message).init(allocator);
        for (sess.messages.items) |msg| {
            try history.append(try msg.clone(allocator));
        }
        return .{ .sid = sid, .history = history };
    }

    const now = nowSeconds();
    const key = try global_allocator.dupe(u8, sid);
    const sess = Session.init(global_allocator, now);
    try sessions_map.put(key, sess);

    const history = std.ArrayList(Message).init(allocator);
    return .{ .sid = sid, .history = history };
}

fn handleChatEndpoint(allocator: std.mem.Allocator, conn_in: std.net.Server.Connection, body: []const u8) !void {
    var conn = conn_in;
    const writer = conn.stream.writer();

    if (hpc_ai_api_key.len == 0) {
        const resp = "{\"detail\":\"AI client not configured\"}";
        try sendHttpResponse(writer, 503, "Service Unavailable", "application/json", &[_][2][]const u8{}, resp);
        return;
    }

    var req_data = parseChatRequest(allocator, body) catch {
        const resp = "{\"detail\":\"Invalid request\"}";
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    defer req_data.deinit(allocator);

    const session_result = getOrCreateSession(allocator, req_data.session_id) catch {
        const resp = "{\"detail\":\"Session error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    const sid = session_result.sid;
    defer allocator.free(sid);
    var history = session_result.history;
    defer {
        for (history.items) |*msg| msg.deinit(allocator);
        history.deinit();
    }

    const session_lock = getOrCreateSessionLock(sid) catch {
        const resp = "{\"detail\":\"Internal error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };

    const msg_id = generateUuid(allocator) catch {
        const resp = "{\"detail\":\"Internal error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    defer allocator.free(msg_id);

    try sendSseHeaders(writer);

    const session_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"session\",\"session_id\":\"{s}\"}}", .{sid});
    defer allocator.free(session_ev);
    try sendSseEvent(writer, session_ev);

    session_lock.lock();
    defer session_lock.unlock();

    if (isDeleted(sid)) {
        const err_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"error\",\"message\":\"Session was deleted\"}}", .{});
        defer allocator.free(err_ev);
        try sendSseEvent(writer, err_ev);
        const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
        defer allocator.free(done_ev);
        try sendSseEvent(writer, done_ev);
        return;
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (!sessions_map.contains(sid)) {
            const key = try global_allocator.dupe(u8, sid);
            const now = nowSeconds();
            var sess = Session.init(global_allocator, now);
            for (history.items) |msg| {
                try sess.messages.append(try msg.clone(global_allocator));
            }
            try sessions_map.put(key, sess);
        }
        if (sessions_map.getPtr(sid)) |sess| {
            const user_msg = Message{
                .role = try global_allocator.dupe(u8, "user"),
                .content = MessageContent{ .text = try global_allocator.dupe(u8, req_data.message) },
                .tool_calls = null,
                .tool_call_id = null,
                .msg_id = try global_allocator.dupe(u8, msg_id),
                .agent_mode = req_data.agent_mode,
                .cached_size = null,
            };
            try sess.messages.append(user_msg);
            try trimHistory(global_allocator, &sess.messages);
            sess.updated_at = nowSeconds();
        }
    }
    saveSessionsToDisk();

    var iteration: usize = 0;
    while (iteration < MAX_TOOL_ITERATIONS) : (iteration += 1) {
        if (isDeleted(sid)) {
            const err_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"error\",\"message\":\"Session was deleted\"}}", .{});
            defer allocator.free(err_ev);
            try sendSseEvent(writer, err_ev);
            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        }

        var api_history: std.ArrayList(Message) = blk: {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.getPtr(sid)) |sess| {
                break :blk try buildApiMessages(allocator, sess.messages.items);
            }
            break :blk std.ArrayList(Message).init(allocator);
        };
        defer {
            for (api_history.items) |*msg| msg.deinit(allocator);
            api_history.deinit();
        }

        const sys_prompt = try getSystemPrompt(allocator);
        defer allocator.free(sys_prompt);

        var all_messages = std.ArrayList(Message).init(allocator);
        defer {
            for (all_messages.items) |*msg| msg.deinit(allocator);
            all_messages.deinit();
        }

        try all_messages.append(Message{
            .role = try allocator.dupe(u8, "system"),
            .content = MessageContent{ .text = try allocator.dupe(u8, sys_prompt) },
            .tool_calls = null,
            .tool_call_id = null,
            .msg_id = null,
            .agent_mode = false,
            .cached_size = null,
        });
        for (api_history.items) |msg| {
            try all_messages.append(try msg.clone(allocator));
        }

        const max_tokens = safeMaxTokens(api_history.items);
        const request_body = try buildOpenAIRequest(allocator, "zai-org/glm-5.1", all_messages.items, max_tokens, true, true);
        defer allocator.free(request_body);

        var tool_call_chunks = std.AutoHashMap(usize, *ToolCallAccumulator).init(allocator);
        defer {
            var it = tool_call_chunks.iterator();
            while (it.next()) |entry| entry.value_ptr.*.deinit();
            tool_call_chunks.deinit();
        }

        var finish_reason: ?[]u8 = null;
        defer if (finish_reason) |fr| allocator.free(fr);
        var turn_content = std.ArrayList(u8).init(allocator);
        defer turn_content.deinit();
        var reasoning_started = false;
        var reasoning_done = false;

        var sse_writer = SseWriter{ .conn = &conn, .allocator = allocator };
        var ctx = ChatStreamContext{
            .sse = &sse_writer,
            .tool_call_chunks = &tool_call_chunks,
            .finish_reason = &finish_reason,
            .turn_content = &turn_content,
            .reasoning_started = &reasoning_started,
            .reasoning_done = &reasoning_done,
            .allocator = allocator,
        };

        const base_url = "https://api.hpc-ai.com/inference/v1";
        callOpenAIStreaming(allocator, base_url, hpc_ai_api_key, request_body, chatStreamCallback, &ctx) catch |err| {
            const err_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"error\",\"message\":\"{}\"}}", .{err});
            defer allocator.free(err_ev);
            try sendSseEvent(writer, err_ev);
            rollbackSessionTurn(sid, msg_id);
            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        };

        if (reasoning_started and !reasoning_done) {
            try sendSseEvent(writer, "{\"type\":\"reasoning_end\"}");
        }

        {
            var tcc_it = tool_call_chunks.iterator();
            while (tcc_it.next()) |entry| {
                const acc = entry.value_ptr.*;
                const full_args = try acc.getArguments();
                for (acc.arg_parts.items) |part| allocator.free(part);
                acc.arg_parts.clearRetainingCapacity();
                try acc.arg_parts.append(full_args);
            }
        }

        const is_tool_call = if (finish_reason) |fr| std.mem.eql(u8, fr, "tool_calls") else false;

        if (is_tool_call and tool_call_chunks.count() > 0) {
            var sorted_indices = std.ArrayList(usize).init(allocator);
            defer sorted_indices.deinit();
            var it = tool_call_chunks.keyIterator();
            while (it.next()) |k| try sorted_indices.append(k.*);
            std.sort.block(usize, sorted_indices.items, {}, std.sort.asc(usize));

            var tool_calls_list = std.ArrayList(ToolCall).init(allocator);
            defer {
                for (tool_calls_list.items) |*tc| tc.deinit(allocator);
                tool_calls_list.deinit();
            }

            for (sorted_indices.items) |idx| {
                const acc = tool_call_chunks.get(idx).?;
                const resolved_id = if (acc.id.len > 0) try allocator.dupe(u8, acc.id) else blk2: {
                    const short = try generateShortHex(allocator);
                    const tc_id_str = try std.fmt.allocPrint(allocator, "tc_{s}", .{short});
                    allocator.free(short);
                    break :blk2 tc_id_str;
                };

                const args_str = if (acc.arg_parts.items.len > 0) try allocator.dupe(u8, acc.arg_parts.items[0]) else try allocator.dupe(u8, "{}");

                if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed_check| {
                    parsed_check.deinit();
                } else |_| {
                    allocator.free(args_str);
                    const raw_arg = if (acc.arg_parts.items.len > 0) acc.arg_parts.items[0] else "";
                    const fallback = try std.fmt.allocPrint(allocator, "{{\"raw_invalid_arguments_fallback\":\"{s}\"}}", .{raw_arg});
                    try tool_calls_list.append(ToolCall{
                        .id = resolved_id,
                        .type = try allocator.dupe(u8, "function"),
                        .function = .{
                            .name = try allocator.dupe(u8, acc.name),
                            .arguments = fallback,
                        },
                    });
                    continue;
                }

                try tool_calls_list.append(ToolCall{
                    .id = resolved_id,
                    .type = try allocator.dupe(u8, "function"),
                    .function = .{
                        .name = try allocator.dupe(u8, acc.name),
                        .arguments = args_str,
                    },
                });
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        var tcs_clone = std.ArrayList(ToolCall).init(global_allocator);
                        for (tool_calls_list.items) |tc| {
                            try tcs_clone.append(try tc.clone(global_allocator));
                        }
                        const asst_msg = Message{
                            .role = try global_allocator.dupe(u8, "assistant"),
                            .content = MessageContent{ .text = try global_allocator.dupe(u8, turn_content.items) },
                            .tool_calls = tcs_clone,
                            .tool_call_id = null,
                            .msg_id = null,
                            .agent_mode = false,
                            .cached_size = null,
                        };
                        try sess.messages.append(asst_msg);
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }
            saveSessionsToDisk();

            var has_search_tool = false;
            for (tool_calls_list.items) |tc| {
                if (std.mem.eql(u8, tc.function.name, "exa_search")) {
                    has_search_tool = true;
                    const tool_result = executeExaToolCall(allocator, tc.id, tc.function.arguments, sid) catch |err| blk3: {
                        var evs = std.ArrayList([]u8).init(allocator);
                        const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"search_error\",\"id\":\"{s}\",\"error\":\"{}\"}}", .{ tc.id, err });
                        try evs.append(ev);
                        const tool_msg = Message{
                            .role = try allocator.dupe(u8, "tool"),
                            .content = MessageContent{ .text = try std.fmt.allocPrint(allocator, "Search failed: {}", .{err}) },
                            .tool_calls = null,
                            .tool_call_id = try allocator.dupe(u8, tc.id),
                            .msg_id = null,
                            .agent_mode = false,
                            .cached_size = null,
                        };
                        break :blk3 ExaToolResult{ .events = evs, .tool_msg = tool_msg, .tool_type = "search" };
                    };
                    var tool_msg = tool_result.tool_msg;
                    defer tool_msg.deinit(allocator);

                    for (tool_result.events.items) |ev| {
                        try sendSseEvent(writer, ev);
                        allocator.free(ev);
                    }
                    var events_list = tool_result.events;
                    events_list.deinit();

                    {
                        sessions_mutex.lock();
                        defer sessions_mutex.unlock();
                        if (sessions_map.getPtr(sid)) |sess| {
                            if (!isDeleted(sid)) {
                                try sess.messages.append(try tool_msg.clone(global_allocator));
                                try trimHistory(global_allocator, &sess.messages);
                                sess.updated_at = nowSeconds();
                            }
                        }
                    }
                } else {
                    const ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"tool_error\",\"id\":\"{s}\",\"error\":\"Unknown tool: {s}\"}}", .{ tc.id, tc.function.name });
                    defer allocator.free(ev);
                    try sendSseEvent(writer, ev);

                    const tool_msg = Message{
                        .role = try global_allocator.dupe(u8, "tool"),
                        .content = MessageContent{ .text = try std.fmt.allocPrint(global_allocator, "Error: unknown tool '{s}'", .{tc.function.name}) },
                        .tool_calls = null,
                        .tool_call_id = try global_allocator.dupe(u8, tc.id),
                        .msg_id = null,
                        .agent_mode = false,
                        .cached_size = null,
                    };
                    sessions_mutex.lock();
                    if (sessions_map.getPtr(sid)) |sess| {
                        sess.messages.append(tool_msg) catch {};
                    }
                    sessions_mutex.unlock();
                }
            }

            if (has_search_tool) {
                try sendSseEvent(writer, "{\"type\":\"search_done\"}");
            }
            saveSessionsToDisk();
        } else {
            var final_content = turn_content.items;
            var extra_content: ?[]u8 = null;
            defer if (extra_content) |ec| allocator.free(ec);

            if (finish_reason != null and std.mem.eql(u8, finish_reason.?, "length") and final_content.len > 0) {
                const trunc_msg = "\n\n[Response truncated due to length limit]";
                extra_content = try std.fmt.allocPrint(allocator, "{s}{s}", .{ final_content, trunc_msg });
                final_content = extra_content.?;
                try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"\\n\\n[Response truncated due to length limit]\"}");
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        if (final_content.len > 0) {
                            const asst_msg = Message{
                                .role = try global_allocator.dupe(u8, "assistant"),
                                .content = MessageContent{ .text = try global_allocator.dupe(u8, final_content) },
                                .tool_calls = null,
                                .tool_call_id = null,
                                .msg_id = null,
                                .agent_mode = false,
                                .cached_size = null,
                            };
                            try sess.messages.append(asst_msg);
                        }
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }
            saveSessionsToDisk();

            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        }
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (sessions_map.getPtr(sid)) |sess| {
            if (!isDeleted(sid)) {
                const asst_msg = Message{
                    .role = try global_allocator.dupe(u8, "assistant"),
                    .content = MessageContent{ .text = try global_allocator.dupe(u8, "Tool iteration limit reached. Please try your question again.") },
                    .tool_calls = null,
                    .tool_call_id = null,
                    .msg_id = null,
                    .agent_mode = false,
                    .cached_size = null,
                };
                try sess.messages.append(asst_msg);
                try trimHistory(global_allocator, &sess.messages);
                sess.updated_at = nowSeconds();
            }
        }
    }
    saveSessionsToDisk();

    try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"Tool iteration limit reached. Please try your question again.\"}");
    const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
    defer allocator.free(done_ev);
    try sendSseEvent(writer, done_ev);
}

const AgentStreamContext = struct {
    sse_writer: *const std.net.Server.Connection,
    tool_call_chunks: *std.AutoHashMap(usize, *ToolCallAccumulator),
    finish_reason: *?[]u8,
    turn_content: *std.ArrayList(u8),
    reasoning_started: *bool,
    reasoning_done: *bool,
    allocator: std.mem.Allocator,
};

fn agentStreamCallback(allocator: std.mem.Allocator, line: []const u8, ctx: *AgentStreamContext) !void {
    const chunk = parseSseLine(allocator, line) catch return;
    var c = chunk orelse return;
    defer c.deinit();

    if (c.finish_reason) |fr| {
        if (ctx.finish_reason.*) |old| allocator.free(old);
        ctx.finish_reason.* = try allocator.dupe(u8, fr);
    }

    if (c.reasoning_content) |r| {
        if (!ctx.reasoning_started.*) {
            ctx.reasoning_started.* = true;
            try ctx.sse_writer.stream.writeAll("data: {\"type\":\"reasoning_start\"}\n\n");
        }
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"reasoning\",\"text\":");
        try writeJsonString(w, r);
        try w.writeByte('}');
        try sendSseEvent(ctx.sse_writer.stream.writer(), ev_buf.items);
    }

    if (c.content) |content| {
        if (ctx.reasoning_started.* and !ctx.reasoning_done.*) {
            ctx.reasoning_done.* = true;
            try ctx.sse_writer.stream.writeAll("data: {\"type\":\"reasoning_end\"}\n\n");
        }
        try ctx.turn_content.appendSlice(content);
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"content\",\"text\":");
        try writeJsonString(w, content);
        try w.writeByte('}');
        try sendSseEvent(ctx.sse_writer.stream.writer(), ev_buf.items);
    }

    if (c.tool_calls) |tcs| {
        for (tcs.items) |tc| {
            const idx = tc.index orelse continue;
            if (ctx.tool_call_chunks.get(idx) == null) {
                const acc = try ToolCallAccumulator.init(allocator);
                try ctx.tool_call_chunks.put(idx, acc);
            }
            const acc = ctx.tool_call_chunks.get(idx).?;
            if (tc.id) |id| {
                allocator.free(acc.id);
                acc.id = try allocator.dupe(u8, id);
            }
            if (tc.name) |n| {
                allocator.free(acc.name);
                acc.name = try allocator.dupe(u8, n);
            }
            if (tc.arguments) |a| {
                try acc.arg_parts.append(try allocator.dupe(u8, a));
            }
        }
    }
}

fn handleAgentEndpoint(allocator: std.mem.Allocator, conn_in: std.net.Server.Connection, body: []const u8) !void {
    var conn = conn_in;
    const writer = conn.stream.writer();

    if (fireworks_api_key.len == 0) {
        const resp = "{\"detail\":\"Agent not configured\"}";
        try sendHttpResponse(writer, 503, "Service Unavailable", "application/json", &[_][2][]const u8{}, resp);
        return;
    }

    var req_data = parseChatRequest(allocator, body) catch {
        const resp = "{\"detail\":\"Invalid request\"}";
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    defer req_data.deinit(allocator);

    const session_result = getOrCreateSession(allocator, req_data.session_id) catch {
        const resp = "{\"detail\":\"Session error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    const sid = session_result.sid;
    defer allocator.free(sid);
    var history = session_result.history;
    defer {
        for (history.items) |*msg| msg.deinit(allocator);
        history.deinit();
    }

    const session_lock = getOrCreateSessionLock(sid) catch {
        const resp = "{\"detail\":\"Internal error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };

    const msg_id = generateUuid(allocator) catch {
        const resp = "{\"detail\":\"Internal error\"}";
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, resp);
        return;
    };
    defer allocator.free(msg_id);

    try sendSseHeaders(writer);

    const session_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"session\",\"session_id\":\"{s}\"}}", .{sid});
    defer allocator.free(session_ev);
    try sendSseEvent(writer, session_ev);

    session_lock.lock();
    defer session_lock.unlock();

    if (isDeleted(sid)) {
        try sendSseEvent(writer, "{\"type\":\"error\",\"message\":\"Session was deleted\"}");
        const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
        defer allocator.free(done_ev);
        try sendSseEvent(writer, done_ev);
        return;
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (!sessions_map.contains(sid)) {
            const key = try global_allocator.dupe(u8, sid);
            const now = nowSeconds();
            var sess = Session.init(global_allocator, now);
            for (history.items) |msg| {
                try sess.messages.append(try msg.clone(global_allocator));
            }
            try sessions_map.put(key, sess);
        }
        if (sessions_map.getPtr(sid)) |sess| {
            const user_msg = Message{
                .role = try global_allocator.dupe(u8, "user"),
                .content = MessageContent{ .text = try global_allocator.dupe(u8, req_data.message) },
                .tool_calls = null,
                .tool_call_id = null,
                .msg_id = try global_allocator.dupe(u8, msg_id),
                .agent_mode = true,
                .cached_size = null,
            };
            try sess.messages.append(user_msg);
            try trimHistory(global_allocator, &sess.messages);
            sess.updated_at = nowSeconds();
        }
    }
    saveSessionsToDisk();

    var iteration: usize = 0;
    while (iteration < MAX_TOOL_ITERATIONS) : (iteration += 1) {
        if (isDeleted(sid)) {
            try sendSseEvent(writer, "{\"type\":\"error\",\"message\":\"Session was deleted\"}");
            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        }

        var api_history: std.ArrayList(Message) = blk: {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.getPtr(sid)) |sess| {
                break :blk try buildApiMessages(allocator, sess.messages.items);
            }
            break :blk std.ArrayList(Message).init(allocator);
        };
        defer {
            for (api_history.items) |*msg| msg.deinit(allocator);
            api_history.deinit();
        }

        var clean_history = std.ArrayList(Message).init(allocator);
        defer {
            for (clean_history.items) |*msg| msg.deinit(allocator);
            clean_history.deinit();
        }

        for (api_history.items) |msg| {
            var combined = std.ArrayList(u8).init(allocator);
            defer combined.deinit();
            switch (msg.content) {
                .text => |t| try combined.appendSlice(t),
                .parts => |parts| {
                    for (parts.items) |part| {
                        if (std.mem.eql(u8, part.type, "text")) {
                            if (part.text) |t| try combined.appendSlice(t);
                        } else if (std.mem.eql(u8, part.type, "image_url")) {
                            try combined.appendSlice(" [Image attachment present]");
                        }
                    }
                },
            }
            try clean_history.append(Message{
                .role = try allocator.dupe(u8, msg.role),
                .content = MessageContent{ .text = try combined.toOwnedSlice() },
                .tool_calls = if (msg.tool_calls) |tcs| blk2: {
                    var new_tcs = std.ArrayList(ToolCall).init(allocator);
                    for (tcs.items) |tc| try new_tcs.append(try tc.clone(allocator));
                    break :blk2 new_tcs;
                } else null,
                .tool_call_id = if (msg.tool_call_id) |id| try allocator.dupe(u8, id) else null,
                .msg_id = null,
                .agent_mode = false,
                .cached_size = null,
            });
        }

        var all_messages = std.ArrayList(Message).init(allocator);
        defer {
            for (all_messages.items) |*msg| msg.deinit(allocator);
            all_messages.deinit();
        }

        try all_messages.append(Message{
            .role = try allocator.dupe(u8, "system"),
            .content = MessageContent{ .text = try allocator.dupe(u8, AGENT_SYSTEM_PROMPT) },
            .tool_calls = null,
            .tool_call_id = null,
            .msg_id = null,
            .agent_mode = false,
            .cached_size = null,
        });
        for (clean_history.items) |msg| {
            try all_messages.append(try msg.clone(allocator));
        }

        const max_tokens = safeMaxTokens(clean_history.items);
        const request_body = try buildAgentOpenAIRequest(allocator, "accounts/fireworks/models/glm-5", all_messages.items, max_tokens);
        defer allocator.free(request_body);

        var tool_call_chunks = std.AutoHashMap(usize, *ToolCallAccumulator).init(allocator);
        defer {
            var it = tool_call_chunks.iterator();
            while (it.next()) |entry| entry.value_ptr.*.deinit();
            tool_call_chunks.deinit();
        }

        var finish_reason: ?[]u8 = null;
        defer if (finish_reason) |fr| allocator.free(fr);
        var turn_content = std.ArrayList(u8).init(allocator);
        defer turn_content.deinit();
        var reasoning_started = false;
        var reasoning_done = false;

        var agent_ctx = AgentStreamContext{
            .sse_writer = &conn,
            .tool_call_chunks = &tool_call_chunks,
            .finish_reason = &finish_reason,
            .turn_content = &turn_content,
            .reasoning_started = &reasoning_started,
            .reasoning_done = &reasoning_done,
            .allocator = allocator,
        };

        const base_url = "https://api.fireworks.ai/inference/v1";
        callOpenAIStreaming(allocator, base_url, fireworks_api_key, request_body, agentStreamCallback, &agent_ctx) catch |err| {
            const err_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"error\",\"message\":\"{}\"}}", .{err});
            defer allocator.free(err_ev);
            try sendSseEvent(writer, err_ev);
            rollbackSessionTurn(sid, msg_id);
            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        };

        if (reasoning_started and !reasoning_done) {
            try sendSseEvent(writer, "{\"type\":\"reasoning_end\"}");
        }

        {
            var tcc_it = tool_call_chunks.iterator();
            while (tcc_it.next()) |entry| {
                const acc = entry.value_ptr.*;
                const full_args = try acc.getArguments();
                for (acc.arg_parts.items) |part| allocator.free(part);
                acc.arg_parts.clearRetainingCapacity();
                try acc.arg_parts.append(full_args);
            }
        }

        const is_tool_call = if (finish_reason) |fr| std.mem.eql(u8, fr, "tool_calls") else false;

        if (is_tool_call and tool_call_chunks.count() > 0) {
            var sorted_indices = std.ArrayList(usize).init(allocator);
            defer sorted_indices.deinit();
            var it = tool_call_chunks.keyIterator();
            while (it.next()) |k| try sorted_indices.append(k.*);
            std.sort.block(usize, sorted_indices.items, {}, std.sort.asc(usize));

            var tool_calls_list = std.ArrayList(ToolCall).init(allocator);
            defer {
                for (tool_calls_list.items) |*tc| tc.deinit(allocator);
                tool_calls_list.deinit();
            }

            for (sorted_indices.items) |idx| {
                const acc = tool_call_chunks.get(idx).?;
                const resolved_id = if (acc.id.len > 0) try allocator.dupe(u8, acc.id) else blk2: {
                    const short = try generateShortHex(allocator);
                    const tc_id_str = try std.fmt.allocPrint(allocator, "tc_{s}", .{short});
                    allocator.free(short);
                    break :blk2 tc_id_str;
                };
                const args_str = if (acc.arg_parts.items.len > 0) try allocator.dupe(u8, acc.arg_parts.items[0]) else try allocator.dupe(u8, "{}");

                try tool_calls_list.append(ToolCall{
                    .id = resolved_id,
                    .type = try allocator.dupe(u8, "function"),
                    .function = .{
                        .name = try allocator.dupe(u8, acc.name),
                        .arguments = args_str,
                    },
                });
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        var tcs_clone = std.ArrayList(ToolCall).init(global_allocator);
                        for (tool_calls_list.items) |tc| {
                            try tcs_clone.append(try tc.clone(global_allocator));
                        }
                        const asst_msg = Message{
                            .role = try global_allocator.dupe(u8, "assistant"),
                            .content = MessageContent{ .text = try global_allocator.dupe(u8, turn_content.items) },
                            .tool_calls = tcs_clone,
                            .tool_call_id = null,
                            .msg_id = null,
                            .agent_mode = true,
                            .cached_size = null,
                        };
                        try sess.messages.append(asst_msg);
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }
            saveSessionsToDisk();

            for (tool_calls_list.items) |tc| {
                const tool_name = tc.function.name;
                const tool_args = tc.function.arguments;
                const tc_id = tc.id;

                var ev_buf = std.ArrayList(u8).init(allocator);
                const ew = ev_buf.writer();
                try ew.writeAll("{\"type\":\"tool_call\",\"name\":");
                try writeJsonString(ew, tool_name);
                try ew.writeAll(",\"id\":");
                try writeJsonString(ew, tc_id);
                try ew.writeByte('}');
                const tool_call_ev = try ev_buf.toOwnedSlice();
                defer allocator.free(tool_call_ev);
                try sendSseEvent(writer, tool_call_ev);

                const tool_result = executeAgentTool(allocator, tool_name, tool_args) catch |err| blk3: {
                    const evs = std.ArrayList([]u8).init(allocator);
                    break :blk3 AgentToolResult{
                        .result = try std.fmt.allocPrint(allocator, "{{\"error\":\"{}\"}}", .{err}),
                        .events = evs,
                    };
                };
                defer allocator.free(tool_result.result);

                for (tool_result.events.items) |ev| {
                    try sendSseEvent(writer, ev);
                    allocator.free(ev);
                }
                var events_list = tool_result.events;
                events_list.deinit();

                {
                    sessions_mutex.lock();
                    defer sessions_mutex.unlock();
                    if (sessions_map.getPtr(sid)) |sess| {
                        if (!isDeleted(sid)) {
                            const tool_msg = Message{
                                .role = try global_allocator.dupe(u8, "tool"),
                                .content = MessageContent{ .text = try global_allocator.dupe(u8, tool_result.result) },
                                .tool_calls = null,
                                .tool_call_id = try global_allocator.dupe(u8, tc_id),
                                .msg_id = null,
                                .agent_mode = false,
                                .cached_size = null,
                            };
                            try sess.messages.append(tool_msg);
                            try trimHistory(global_allocator, &sess.messages);
                            sess.updated_at = nowSeconds();
                        }
                    }
                }
            }
            saveSessionsToDisk();
        } else {
            var final_content = turn_content.items;
            var extra_content: ?[]u8 = null;
            defer if (extra_content) |ec| allocator.free(ec);

            if (finish_reason != null and std.mem.eql(u8, finish_reason.?, "length") and final_content.len > 0) {
                extra_content = try std.fmt.allocPrint(allocator, "{s}\n\n[Response truncated due to length limit]", .{final_content});
                final_content = extra_content.?;
                try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"\\n\\n[Response truncated due to length limit]\"}");
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        if (final_content.len > 0) {
                            const asst_msg = Message{
                                .role = try global_allocator.dupe(u8, "assistant"),
                                .content = MessageContent{ .text = try global_allocator.dupe(u8, final_content) },
                                .tool_calls = null,
                                .tool_call_id = null,
                                .msg_id = null,
                                .agent_mode = true,
                                .cached_size = null,
                            };
                            try sess.messages.append(asst_msg);
                        }
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }
            saveSessionsToDisk();

            const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
            defer allocator.free(done_ev);
            try sendSseEvent(writer, done_ev);
            return;
        }
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (sessions_map.getPtr(sid)) |sess| {
            if (!isDeleted(sid)) {
                const asst_msg = Message{
                    .role = try global_allocator.dupe(u8, "assistant"),
                    .content = MessageContent{ .text = try global_allocator.dupe(u8, "Tool iteration limit reached. Please try your question again.") },
                    .tool_calls = null,
                    .tool_call_id = null,
                    .msg_id = null,
                    .agent_mode = true,
                    .cached_size = null,
                };
                try sess.messages.append(asst_msg);
                try trimHistory(global_allocator, &sess.messages);
                sess.updated_at = nowSeconds();
            }
        }
    }
    saveSessionsToDisk();

    try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"Tool iteration limit reached. Please try your question again.\"}");
    const done_ev = try std.fmt.allocPrint(allocator, "{{\"type\":\"done\",\"session_id\":\"{s}\"}}", .{sid});
    defer allocator.free(done_ev);
    try sendSseEvent(writer, done_ev);
}

pub fn main() !void {
    global_allocator = gpa.allocator();

    hpc_ai_api_key = try getEnvAlloc(global_allocator, "HPC_AI_API_KEY", "");
    exa_api_key = try getEnvAlloc(global_allocator, "EXA_API_KEY", "");
    fireworks_api_key = try getEnvAlloc(global_allocator, "FIREWORKS_API_KEY", "");
    obj_bucket = try getEnvAlloc(global_allocator, "OBJ_BUCKET", "");
    server_host = try getEnvAlloc(global_allocator, "HOST", "0.0.0.0");

    const port_str = try getEnvAlloc(global_allocator, "PORT", "5000");
    defer global_allocator.free(port_str);
    server_port = std.fmt.parseInt(u16, port_str, 10) catch 5000;

    const project_root_env = try getEnvAlloc(global_allocator, "PROJECT_ROOT", "");
    if (project_root_env.len > 0) {
        project_root = project_root_env;
    } else {
        global_allocator.free(project_root_env);
        const cwd_path = std.fs.cwd().realpathAlloc(global_allocator, ".") catch blk: {
            const exe_dir = try std.fs.selfExeDirPathAlloc(global_allocator);
            break :blk exe_dir;
        };
        project_root = cwd_path;
    }

    index_file = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "index.html" });
    static_dir = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "static" });
    static_sw_file = try std.fs.path.join(global_allocator, &[_][]const u8{ static_dir, "sw.js" });
    persistence_file = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "sessions_db.json" });

    sessions_map = std.StringHashMap(Session).init(global_allocator);
    session_locks_map = std.StringHashMap(*std.Thread.Mutex).init(global_allocator);
    deleted_sessions_map = std.StringHashMap(f64).init(global_allocator);
    session_exa_limiters_map = std.StringHashMap(*RateLimiter).init(global_allocator);
    exa_limiter = RateLimiter.init(global_allocator, EXA_QPS_LIMIT);

    loadSessionsFromDisk();

    std.log.info("Starting Helium server on {s}:{d}", .{ server_host, server_port });

    const addr = try std.net.Address.parseIp(server_host, server_port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.log.info("Helium server running on port {d}", .{server_port});

    while (true) {
        const conn = server.accept() catch |err| {
            std.log.err("Accept error: {}", .{err});
            continue;
        };

        const thread_alloc = global_allocator;
        const ctx = ConnectionContext{ .conn = conn, .allocator = thread_alloc };
        const thread = std.Thread.spawn(.{}, handleConnection, .{ctx}) catch |err| {
            std.log.err("Failed to spawn thread: {}", .{err});
            conn.stream.close();
            continue;
        };
        thread.detach();
    }
}
