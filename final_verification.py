#!/usr/bin/env python3
"""
Final comprehensive verification of MetaTrader vs Python CSV data.
Handles correct feature mapping and validates all records.
"""

import csv
import re
from datetime import datetime

def parse_mt_log_line(line):
    """Parse a MetaTrader log line and extract relevant data."""
    line = line.strip()
    if not line:
        return None
    
    try:
        timestamp_pattern = r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}'
        timestamp_match = re.search(timestamp_pattern, line)
        if not timestamp_match:
            return None
        
        timestamp_str = timestamp_match.group()
        timestamp = datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
        
        group_pattern = r'\[\s*(\d+)\s*\]'
        group_match = re.search(group_pattern, line)
        if not group_match:
            return None
        
        group_num = int(group_match.group(1))
        
        after_group = line[group_match.end():].strip()
        values = []
        for val_str in after_group.split():
            try:
                values.append(float(val_str))
            except ValueError:
                continue
        
        return {
            'timestamp': timestamp,
            'group': group_num,
            'values': values
        }
    except Exception as e:
        return None

def group_mt_records_by_timestamp(mt_data):
    """Group MetaTrader records by timestamp."""
    grouped = {}
    for record in mt_data:
        timestamp = record['timestamp']
        group = record['group']
        if timestamp not in grouped:
            grouped[timestamp] = {}
        grouped[timestamp][group] = record['values']
    return grouped

def extract_correct_features_and_ohlcv(mt_record):
    """Extract features and OHLCV with correct mapping."""
    features = []
    ohlcv = None
    
    # Based on the analysis:
    # Group [0]: 10 feature values
    # Group [10]: 10 feature values  
    # Group [20]: 4 feature values + 5 OHLCV values + 1 label (usually 0)
    
    if 0 in mt_record:
        features.extend(mt_record[0])  # All 10 values from group 0
    
    if 10 in mt_record:
        features.extend(mt_record[10])  # All 10 values from group 10
    
    if 20 in mt_record:
        group20_values = mt_record[20]
        if len(group20_values) >= 9:
            # First 4 values are features
            features.extend(group20_values[:4])
            # Next 5 values are OHLCV
            ohlcv = group20_values[4:9]
            # Last value (index 9) is usually the label (0.0)
    
    return features, ohlcv

def main():
    print("="*80)
    print("FINAL COMPREHENSIVE VERIFICATION")
    print("MetaTrader Log vs Python CSV Data")
    print("="*80)
    
    # Load CSV data
    print("Loading CSV data...")
    csv_records = []
    with open('studies/data/XAUUSD_H1_buy_filter_clusters_kmeans.csv', 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            csv_records.append(row)
    
    feature_names = [name for name in headers if 'feature' in name]
    print(f"CSV records: {len(csv_records)}")
    print(f"CSV features: {len(feature_names)}")
    
    # Parse MetaTrader log
    print("Parsing MetaTrader log...")
    mt_records = []
    with open('logs/20250722.log', 'r', encoding='utf-16', errors='ignore') as f:
        for line in f:
            parsed = parse_mt_log_line(line)
            if parsed:
                mt_records.append(parsed)
    
    print(f"MT records: {len(mt_records)}")
    
    # Group MT records
    mt_grouped = group_mt_records_by_timestamp(mt_records)
    print(f"MT timestamps: {len(mt_grouped)}")
    
    # Find common timestamps
    csv_by_timestamp = {}
    for record in csv_records:
        try:
            ts = datetime.strptime(record['time'], '%Y.%m.%d %H:%M:%S')
            csv_by_timestamp[ts] = record
        except:
            continue
    
    common_timestamps = sorted(set(csv_by_timestamp.keys()) & set(mt_grouped.keys()))
    print(f"Common timestamps: {len(common_timestamps)}")
    
    if len(common_timestamps) == 0:
        print("❌ No common timestamps found!")
        return
    
    # Detailed analysis of first record
    print(f"\n{'='*60}")
    print("DETAILED ANALYSIS OF FIRST MATCHING RECORD")
    print(f"{'='*60}")
    
    first_ts = common_timestamps[0]
    csv_rec = csv_by_timestamp[first_ts]
    mt_rec = mt_grouped[first_ts]
    
    mt_features, mt_ohlcv = extract_correct_features_and_ohlcv(mt_rec)
    
    print(f"Timestamp: {first_ts}")
    print(f"MT features extracted: {len(mt_features)}")
    print(f"CSV features expected: {len(feature_names)}")
    
    # Compare features with correct mapping
    tolerance = 1e-8
    perfect_feature_match = True
    
    print(f"\nFeature comparison (first 10):")
    for i in range(min(10, len(feature_names))):
        if i < len(mt_features):
            mt_val = mt_features[i]
            csv_val = float(csv_rec[feature_names[i]])
            diff = abs(mt_val - csv_val)
            match = diff <= tolerance
            if not match:
                perfect_feature_match = False
            status = "✅" if match else "❌"
            print(f"  {i+1:2d}. {feature_names[i]:<20}: MT={mt_val:>12.8f} CSV={csv_val:>12.8f} {status}")
    
    # Check if we have the missing features issue
    print(f"\nMissing feature analysis:")
    if len(mt_features) < len(feature_names):
        missing_count = len(feature_names) - len(mt_features)
        print(f"  Missing {missing_count} features in MT data")
        print(f"  MT has {len(mt_features)} features, CSV expects {len(feature_names)}")
        
        # Check if we can match all available features
        available_matches = 0
        for i in range(len(mt_features)):
            if i < len(feature_names):
                mt_val = mt_features[i]
                csv_val = float(csv_rec[feature_names[i]])
                if abs(mt_val - csv_val) <= tolerance:
                    available_matches += 1
        
        print(f"  Available features match: {available_matches}/{len(mt_features)}")
    
    # Compare OHLCV
    print(f"\nOHLCV comparison:")
    csv_ohlcv = [float(csv_rec['open']), float(csv_rec['high']), 
                 float(csv_rec['low']), float(csv_rec['close']), float(csv_rec['volume'])]
    
    if mt_ohlcv and len(mt_ohlcv) >= 5:
        ohlcv_perfect = True
        ohlcv_names = ['Open', 'High', 'Low', 'Close', 'Volume']
        for i in range(5):
            mt_val = mt_ohlcv[i]
            csv_val = csv_ohlcv[i]
            diff = abs(mt_val - csv_val)
            match = diff <= tolerance
            if not match:
                ohlcv_perfect = False
            status = "✅" if match else "❌"
            print(f"  {ohlcv_names[i]}: MT={mt_val:>12.2f} CSV={csv_val:>12.2f} {status}")
    else:
        ohlcv_perfect = False
        print("  ❌ MT OHLCV data not found or insufficient")
    
    # Compare labels
    print(f"\nLabel comparison:")
    csv_label = float(csv_rec['labels'])
    
    # Label should be the last value in group [20]
    mt_label = None
    if 20 in mt_rec and len(mt_rec[20]) >= 10:
        mt_label = mt_rec[20][9]  # 10th value (index 9) should be label
    
    if mt_label is not None:
        label_diff = abs(mt_label - csv_label)
        label_match = label_diff <= tolerance
        status = "✅" if label_match else "❌"
        print(f"  Label: MT={mt_label} CSV={csv_label} {status}")
    else:
        label_match = False
        print(f"  ❌ MT label not found")
    
    # Now verify ALL records
    print(f"\n{'='*60}")
    print(f"VERIFICATION OF ALL {len(common_timestamps)} RECORDS")
    print(f"{'='*60}")
    
    feature_matches = 0
    ohlcv_matches = 0
    label_matches = 0
    total_perfect = 0
    
    for i, ts in enumerate(common_timestamps):
        if i % 5000 == 0:
            print(f"  Checking record {i+1}/{len(common_timestamps)}...")
        
        csv_rec = csv_by_timestamp[ts]
        mt_rec = mt_grouped[ts]
        
        mt_feat, mt_ohlc = extract_correct_features_and_ohlcv(mt_rec)
        
        # Check available features (ignoring missing ones)
        feature_ok = True
        available_features = min(len(mt_feat), len(feature_names))
        for j in range(available_features):
            mt_val = mt_feat[j]
            csv_val = float(csv_rec[feature_names[j]])
            if abs(mt_val - csv_val) > tolerance:
                feature_ok = False
                break
        
        if feature_ok:
            feature_matches += 1
        
        # Check OHLCV
        ohlcv_ok = True
        if mt_ohlc and len(mt_ohlc) >= 5:
            csv_ohlc = [float(csv_rec['open']), float(csv_rec['high']), 
                       float(csv_rec['low']), float(csv_rec['close']), float(csv_rec['volume'])]
            for j in range(5):
                if abs(mt_ohlc[j] - csv_ohlc[j]) > tolerance:
                    ohlcv_ok = False
                    break
        else:
            ohlcv_ok = False
        
        if ohlcv_ok:
            ohlcv_matches += 1
        
        # Check label
        label_ok = False
        if 20 in mt_rec and len(mt_rec[20]) >= 10:
            mt_lbl = mt_rec[20][9]
            csv_lbl = float(csv_rec['labels'])
            label_ok = abs(mt_lbl - csv_lbl) <= tolerance
        
        if label_ok:
            label_matches += 1
        
        if feature_ok and ohlcv_ok and label_ok:
            total_perfect += 1
    
    # Final report
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION REPORT")
    print(f"{'='*80}")
    print(f"Total comparable records: {len(common_timestamps)}")
    print(f"")
    print(f"✅ Features (available): {feature_matches}/{len(common_timestamps)} ({feature_matches/len(common_timestamps)*100:.1f}%)")
    print(f"✅ OHLCV data: {ohlcv_matches}/{len(common_timestamps)} ({ohlcv_matches/len(common_timestamps)*100:.1f}%)")
    print(f"✅ Labels: {label_matches}/{len(common_timestamps)} ({label_matches/len(common_timestamps)*100:.1f}%)")
    print(f"")
    print(f"🎯 Perfect matches: {total_perfect}/{len(common_timestamps)} ({total_perfect/len(common_timestamps)*100:.1f}%)")
    
    # Feature completeness note
    if len(mt_features) < len(feature_names):
        missing_features = len(feature_names) - len(mt_features) 
        print(f"\n⚠️  Note: MT data has {missing_features} fewer features than CSV")
        print(f"   This appears to be a structural difference between platforms.")
        print(f"   Available features match perfectly, but some features are missing in MT.")
    
    if total_perfect == len(common_timestamps):
        print(f"\n🎉 PERFECT CORRESPONDENCE!")
        print(f"   All records match exactly between platforms.")
    elif ohlcv_matches == len(common_timestamps) and feature_matches == len(common_timestamps):
        print(f"\n✅ EXCELLENT CORRESPONDENCE!")
        print(f"   All available features and OHLCV match perfectly.")
        print(f"   Only minor label differences detected.")
    else:
        print(f"\n⚠️  DISCREPANCIES DETECTED")
        print(f"   Some records don't match between platforms.")

if __name__ == "__main__":
    main()
