/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of ExtendedTemporalMemory
 */

#include <cstring>
#include <climits>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>
#include <nupic/experimental/ExtendedTemporalMemory.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::extended_temporal_memory;

#define EPSILON 0.000001

ExtendedTemporalMemory::ExtendedTemporalMemory()
{
  version_ = 2;
}

ExtendedTemporalMemory::ExtendedTemporalMemory(
  vector<UInt> columnDimensions,
  UInt cellsPerColumn,
  UInt activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  UInt minThreshold,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Permanence predictedSegmentDecrement,
  Int seed,
  UInt maxSegmentsPerCell,
  UInt maxSynapsesPerSegment)
{
  initialize(
    columnDimensions,
    cellsPerColumn,
    activationThreshold,
    initialPermanence,
    connectedPermanence,
    minThreshold,
    maxNewSynapseCount,
    permanenceIncrement,
    permanenceDecrement,
    predictedSegmentDecrement,
    seed,
    maxSegmentsPerCell,
    maxSynapsesPerSegment);
}

ExtendedTemporalMemory::~ExtendedTemporalMemory()
{
}

void ExtendedTemporalMemory::initialize(
  vector<UInt> columnDimensions,
  UInt cellsPerColumn,
  UInt activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  UInt minThreshold,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Permanence predictedSegmentDecrement,
  Int seed,
  UInt maxSegmentsPerCell,
  UInt maxSynapsesPerSegment)
{
  // Validate all input parameters

  if (columnDimensions.size() <= 0)
    NTA_THROW << "Number of column dimensions must be greater than 0";

  if (cellsPerColumn <= 0)
    NTA_THROW << "Number of cells per column must be greater than 0";

  NTA_CHECK(initialPermanence >= 0.0 && initialPermanence <= 1.0);
  NTA_CHECK(connectedPermanence >= 0.0 && connectedPermanence <= 1.0);
  NTA_CHECK(permanenceIncrement >= 0.0 && permanenceIncrement <= 1.0);
  NTA_CHECK(permanenceDecrement >= 0.0 && permanenceDecrement <= 1.0);

  // Save member variables

  numColumns_ = 1;
  columnDimensions_.clear();
  for (auto & columnDimension : columnDimensions)
  {
    numColumns_ *= columnDimension;
    columnDimensions_.push_back(columnDimension);
  }

  cellsPerColumn_ = cellsPerColumn;
  activationThreshold_ = activationThreshold;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;
  predictedSegmentDecrement_ = predictedSegmentDecrement;

  // Initialize member variables
  connections = Connections(
    numberOfCells(),
    maxSegmentsPerCell,
    maxSynapsesPerSegment);
  seed_((UInt64)(seed < 0 ? rand() : seed));

  activeCells_.clear();
  activeSegments_.clear();
  winnerCells_.clear();
  matchingSegments_.clear();
}

struct ExcitedColumnData
{
  UInt column;
  bool isActiveColumn;
  vector<SegmentOverlap>::const_iterator activeSegmentsBegin;
  vector<SegmentOverlap>::const_iterator activeSegmentsEnd;
  vector<SegmentOverlap>::const_iterator matchingSegmentsBegin;
  vector<SegmentOverlap>::const_iterator matchingSegmentsEnd;
};

/**
 * Walk the sorted lists of active columns, active segments, and matching
 * segments, grouping them by column. Each list is traversed exactly once.
 *
 * Perform the walk by using iterators.
 */
class ExcitedColumns
{
public:

  ExcitedColumns(const vector<UInt>& activeColumns,
                 const vector<SegmentOverlap>& activeSegments,
                 const vector<SegmentOverlap>& matchingSegments,
                 UInt cellsPerColumn)
    :activeColumns_(activeColumns),
     cellsPerColumn_(cellsPerColumn),
     activeSegments_(activeSegments),
     matchingSegments_(matchingSegments)
  {
    NTA_ASSERT(std::is_sorted(activeColumns.begin(), activeColumns.end()));
    NTA_ASSERT(std::is_sorted(activeSegments.begin(), activeSegments.end(),
                              [](const SegmentOverlap& a,
                                 const SegmentOverlap& b)
                              {
                                return a.segment < b.segment;
                              }));
    NTA_ASSERT(std::is_sorted(matchingSegments.begin(), matchingSegments.end(),
                              [](const SegmentOverlap& a,
                                 const SegmentOverlap& b)
                              {
                                return a.segment < b.segment;
                              }));
  }

  class Iterator
  {
  public:
    Iterator(vector<UInt>::const_iterator activeColumn,
             vector<UInt>::const_iterator activeColumnsEnd,
             vector<SegmentOverlap>::const_iterator activeSegment,
             vector<SegmentOverlap>::const_iterator activeSegmentsEnd,
             vector<SegmentOverlap>::const_iterator matchingSegment,
             vector<SegmentOverlap>::const_iterator matchingSegmentsEnd,
             UInt cellsPerColumn)
      :activeColumn_(activeColumn),
       activeColumnsEnd_(activeColumnsEnd),
       activeSegment_(activeSegment),
       activeSegmentsEnd_(activeSegmentsEnd),
       matchingSegment_(matchingSegment),
       matchingSegmentsEnd_(matchingSegmentsEnd),
       cellsPerColumn_(cellsPerColumn),
       finished_(false)
    {
      calculateNext_();
    }

    bool operator !=(const Iterator& other)
    {
      return finished_ != other.finished_ ||
        activeColumn_ != other.activeColumn_ ||
        activeSegment_ != other.activeSegment_ ||
        matchingSegment_ != other.matchingSegment_;
    }

    const ExcitedColumnData& operator*() const
    {
      NTA_ASSERT(!finished_);
      return current_;
    }

    const Iterator& operator++()
    {
      NTA_ASSERT(!finished_);
      calculateNext_();
      return *this;
    }

  private:

    UInt columnOf_(const SegmentOverlap& segmentOverlap) const
    {
      return segmentOverlap.segment.cell / cellsPerColumn_;
    }

    void calculateNext_()
    {
      if (activeColumn_ != activeColumnsEnd_ ||
          activeSegment_ != activeSegmentsEnd_ ||
          matchingSegment_ != matchingSegmentsEnd_)
      {
        current_.column = UINT_MAX;

        if (activeSegment_ != activeSegmentsEnd_)
        {
          current_.column = std::min(current_.column,
                                     columnOf_(*activeSegment_));
        }

        if (matchingSegment_ != matchingSegmentsEnd_)
        {
          current_.column = std::min(current_.column,
                                     columnOf_(*matchingSegment_));
        }

        if (activeColumn_ != activeColumnsEnd_ &&
            *activeColumn_ <= current_.column)
        {
          current_.column = *activeColumn_;
          current_.isActiveColumn = true;
          activeColumn_++;
        }
        else
        {
          current_.isActiveColumn = false;
        }

        current_.activeSegmentsBegin = activeSegment_;
        while (activeSegment_ != activeSegmentsEnd_ &&
               columnOf_(*activeSegment_) == current_.column)
        {
          activeSegment_++;
        }
        current_.activeSegmentsEnd = activeSegment_;

        current_.matchingSegmentsBegin = matchingSegment_;
        while (matchingSegment_ != matchingSegmentsEnd_ &&
               columnOf_(*matchingSegment_) == current_.column)
        {
          matchingSegment_++;
        }
        current_.matchingSegmentsEnd = matchingSegment_;
      }
      else
      {
        finished_ = true;
      }
    }

    vector<UInt>::const_iterator activeColumn_;
    vector<UInt>::const_iterator activeColumnsEnd_;
    vector<SegmentOverlap>::const_iterator activeSegment_;
    vector<SegmentOverlap>::const_iterator activeSegmentsEnd_;
    vector<SegmentOverlap>::const_iterator matchingSegment_;
    vector<SegmentOverlap>::const_iterator matchingSegmentsEnd_;
    const UInt cellsPerColumn_;

    bool finished_;
    ExcitedColumnData current_;
  };

  Iterator begin()
  {
    return Iterator(activeColumns_.begin(),
                    activeColumns_.end(),
                    activeSegments_.begin(),
                    activeSegments_.end(),
                    matchingSegments_.begin(),
                    matchingSegments_.end(),
                    cellsPerColumn_);
  }

  Iterator end()
  {
    return Iterator(activeColumns_.end(),
                    activeColumns_.end(),
                    activeSegments_.end(),
                    activeSegments_.end(),
                    matchingSegments_.end(),
                    matchingSegments_.end(),
                    cellsPerColumn_);
  }

private:
  const vector<UInt>& activeColumns_;
  const UInt cellsPerColumn_;
  const vector<SegmentOverlap>& activeSegments_;
  const vector<SegmentOverlap>& matchingSegments_;
};

static CellIdx getLeastUsedCell(
  Connections& connections,
  Random& rng,
  UInt column,
  UInt cellsPerColumn)
{
  vector<CellIdx> leastUsedCells;
  UInt32 minNumSegments = UINT_MAX;
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++)
  {
    UInt32 numSegments = connections.segmentsForCell(cell).size();

    if (numSegments < minNumSegments)
    {
      minNumSegments = numSegments;
      leastUsedCells.clear();
    }

    if (numSegments == minNumSegments)
    {
      leastUsedCells.push_back(cell);
    }
  }

  return leastUsedCells[rng.getUInt32(leastUsedCells.size())];
}

static void adaptSegment(
  Connections& connections,
  Segment segment,
  const vector<CellIdx>& prevActiveInternalCells,
  const vector<CellIdx>& prevActiveExternalCells,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  vector<Synapse> synapses = connections.synapsesForSegment(segment);

  for (Synapse synapse : synapses)
  {
    const SynapseData synapseData = connections.dataForSynapse(synapse);

    const bool isActive = (synapseData.presynapticCell < connections.numCells())
      ? std::binary_search(prevActiveInternalCells.begin(),
                           prevActiveInternalCells.end(),
                           synapseData.presynapticCell)
      : std::binary_search(prevActiveExternalCells.begin(),
                           prevActiveExternalCells.end(),
                           synapseData.presynapticCell - connections.numCells());

    Permanence permanence = synapseData.permanence;

    if (isActive)
    {
      permanence += permanenceIncrement;
    }
    else
    {
      permanence -= permanenceDecrement;
    }

    permanence = min(permanence, (Permanence)1.0);
    permanence = max(permanence, (Permanence)0.0);

    if (permanence < EPSILON)
    {
      connections.destroySynapse(synapse);
    }
    else
    {
      connections.updateSynapsePermanence(synapse, permanence);
    }
  }

  if (connections.numSynapses(segment) == 0)
  {
    connections.destroySegment(segment);
  }
}

static void growSynapses(
  Connections& connections,
  Random& rng,
  Segment segment,
  UInt32 nDesiredNewSynapses,
  const vector<CellIdx>& internalCandidates,
  const vector<CellIdx>& externalCandidates,
  Permanence initialPermanence)
{
  vector<CellIdx> candidates;
  candidates.reserve(internalCandidates.size() + externalCandidates.size());
  candidates.insert(candidates.begin(), internalCandidates.begin(),
                    internalCandidates.end());
  for (CellIdx cell : externalCandidates)
  {
    candidates.push_back(cell + connections.numCells());
  }

  // Instead of erasing candidates, swap them to the end, and remember where the
  // "eligible" candidates end.
  auto eligibleEnd = candidates.end();

  // Remove cells that are already synapsed on by this segment
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    CellIdx presynapticCell =
      connections.dataForSynapse(synapse).presynapticCell;
    auto ineligible = find(candidates.begin(), eligibleEnd, presynapticCell);
    if (ineligible != eligibleEnd)
    {
      eligibleEnd--;
      std::iter_swap(ineligible, eligibleEnd);
    }
  }

  const UInt32 nActual =
    std::min(nDesiredNewSynapses,
             (UInt32)std::distance(candidates.begin(), eligibleEnd));

  // Pick nActual cells randomly.
  for (UInt32 c = 0; c < nActual; c++)
  {
    size_t i = rng.getUInt32(std::distance(candidates.begin(), eligibleEnd));;
    connections.createSynapse(segment, candidates[i], initialPermanence);
    eligibleEnd--;
    std::swap(candidates[i], *eligibleEnd);
  }
}

static void activatePredictedColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  const ExcitedColumnData& excitedColumn,
  bool learn,
  const vector<CellIdx>& prevActiveInternalCells,
  const vector<CellIdx>& prevActiveExternalCells,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  auto active = excitedColumn.activeSegmentsBegin;
  do
  {
    const CellIdx cell = active->segment.cell;
    activeCells.push_back(cell);
    winnerCells.push_back(cell);

    // This cell might have multiple active segments.
    do
    {
      if (learn)
      {
        adaptSegment(connections,
                     active->segment,
                     prevActiveInternalCells, prevActiveExternalCells,
                     permanenceIncrement, permanenceDecrement);
      }
      active++;
    } while (active != excitedColumn.activeSegmentsEnd &&
             active->segment.cell == cell);
  } while (active != excitedColumn.activeSegmentsEnd);
}

static void burstColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  Random& rng,
  const ExcitedColumnData& excitedColumn,
  bool learn,
  const vector<CellIdx>& prevActiveInternalCells,
  const vector<CellIdx>& prevActiveExternalCells,
  const vector<CellIdx>& prevWinnerCells,
  UInt cellsPerColumn,
  Permanence initialPermanence,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  const CellIdx start = excitedColumn.column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++)
  {
    activeCells.push_back(cell);
  }

  if (excitedColumn.matchingSegmentsBegin != excitedColumn.matchingSegmentsEnd)
  {
    auto bestMatch = std::max_element(
      excitedColumn.matchingSegmentsBegin,
      excitedColumn.matchingSegmentsEnd,
      [](const SegmentOverlap& a, const SegmentOverlap& b)
      {
        return a.overlap < b.overlap;
      });

    winnerCells.push_back(bestMatch->segment.cell);

    if (learn)
    {
      adaptSegment(connections,
                   bestMatch->segment,
                   prevActiveInternalCells, prevActiveExternalCells,
                   permanenceIncrement, permanenceDecrement);

      const UInt32 nGrowDesired = maxNewSynapseCount - bestMatch->overlap;
      if (nGrowDesired > 0)
      {
        growSynapses(connections, rng,
                     bestMatch->segment, nGrowDesired,
                     prevWinnerCells, prevActiveExternalCells,
                     initialPermanence);
      }
    }
  }
  else
  {
    const CellIdx winnerCell = getLeastUsedCell(connections, rng,
                                                excitedColumn.column,
                                                cellsPerColumn);
    winnerCells.push_back(winnerCell);

    if (learn)
    {
      // Don't grow a segment that will never match.
      const UInt32 nGrowExact =
        std::min(maxNewSynapseCount, (UInt32)(prevWinnerCells.size() +
                                              prevActiveExternalCells.size()));
      if (nGrowExact > 0)
      {
        const Segment segment = connections.createSegment(winnerCell);
        growSynapses(connections, rng,
                     segment, nGrowExact,
                     prevWinnerCells, prevActiveExternalCells,
                     initialPermanence);
        NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
      }
    }
  }
}

static void punishPredictedColumn(
  Connections& connections,
  const ExcitedColumnData& excitedColumn,
  const vector<CellIdx>& prevActiveInternalCells,
  const vector<CellIdx>& prevActiveExternalCells,
  Permanence predictedSegmentDecrement)
{
  if (predictedSegmentDecrement > 0.0)
  {
    for (auto matching = excitedColumn.matchingSegmentsBegin;
         matching != excitedColumn.matchingSegmentsEnd;
         matching++)
    {
      adaptSegment(connections, matching->segment,
                   prevActiveInternalCells, prevActiveExternalCells,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void ExtendedTemporalMemory::activateCells(
  const vector<UInt>& activeColumns,
  const vector<CellIdx>& prevActiveExternalCells,
  bool learn)
{
  NTA_ASSERT(std::is_sorted(activeColumns.begin(), activeColumns.end()));

  const vector<CellIdx> prevActiveInternalCells = std::move(activeCells_);
  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  for (const ExcitedColumnData& excitedColumn : ExcitedColumns(activeColumns,
                                                               activeSegments_,
                                                               matchingSegments_,
                                                               cellsPerColumn_))
  {
    if (excitedColumn.isActiveColumn)
    {
      if (excitedColumn.activeSegmentsBegin != excitedColumn.activeSegmentsEnd)
      {
        activatePredictedColumn(activeCells_, winnerCells_, connections,
                                excitedColumn, learn,
                                prevActiveInternalCells,
                                prevActiveExternalCells,
                                permanenceIncrement_, permanenceDecrement_);
      }
      else
      {
        burstColumn(activeCells_, winnerCells_, connections, rng_,
                    excitedColumn, learn,
                    prevActiveInternalCells, prevActiveExternalCells,
                    prevWinnerCells,
                    cellsPerColumn_, initialPermanence_, maxNewSynapseCount_,
                    permanenceIncrement_, permanenceDecrement_);
      }
    }
    else
    {
      if (learn)
      {
        punishPredictedColumn(connections,
                              excitedColumn,
                              prevActiveInternalCells,
                              prevActiveExternalCells,
                              predictedSegmentDecrement_);
      }
    }
  }
}

void ExtendedTemporalMemory::activateDendrites(
  const vector<CellIdx>& activeExternalCells,
  bool learn)
{
  SegmentExcitationTally excitations(connections, connectedPermanence_, 0.0);
  for (CellIdx cell : activeCells_)
  {
    excitations.addActivePresynapticCell(cell);
  }
  for (CellIdx cell : activeExternalCells)
  {
    excitations.addActivePresynapticCell(cell);
  }

  activeSegments_.clear();
  matchingSegments_.clear();
  excitations.getResults(activationThreshold_, minThreshold_,
                         activeSegments_, matchingSegments_);

  if (learn)
  {
    for (const SegmentOverlap& segmentOverlap : activeSegments_)
    {
      connections.recordSegmentActivity(segmentOverlap.segment);
    }

    connections.startNewIteration();
  }
}

void ExtendedTemporalMemory::compute(
  const vector<UInt>& activeColumnsUnsorted,
  const vector<CellIdx>& prevActiveExternalCells,
  const vector<CellIdx>& activeExternalCells,
  bool learn)
{
  vector<UInt> activeColumns(activeColumnsUnsorted.begin(),
                             activeColumnsUnsorted.end());
  std::sort(activeColumns.begin(), activeColumns.end());

  activateCells(activeColumns, prevActiveExternalCells, learn);
  activateDendrites(activeExternalCells, learn);
}

void ExtendedTemporalMemory::reset(void)
{
  activeCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
  winnerCells_.clear();
}

// ==============================
//  Helper functions
// ==============================

Int ExtendedTemporalMemory::columnForCell(CellIdx cell)
{
  _validateCell(cell);

  return cell / cellsPerColumn_;
}

vector<CellIdx> ExtendedTemporalMemory::cellsForColumn(Int column)
{
  const CellIdx start = cellsPerColumn_ * column;
  const CellIdx end = start + cellsPerColumn_;

  vector<CellIdx> cellsInColumn;
  for (CellIdx i = start; i < end; i++)
  {
    cellsInColumn.push_back(i);
  }

  return cellsInColumn;
}

UInt ExtendedTemporalMemory::numberOfCells(void)
{
  return numberOfColumns() * cellsPerColumn_;
}

vector<CellIdx> ExtendedTemporalMemory::getActiveCells() const
{
  return activeCells_;
}

vector<CellIdx> ExtendedTemporalMemory::getPredictiveCells() const
{
  vector<CellIdx> predictiveCells;

  for (auto segOverlap = activeSegments_.begin();
       segOverlap != activeSegments_.end();
       segOverlap++)
  {
    if (segOverlap == activeSegments_.begin() ||
        segOverlap->segment.cell != predictiveCells.back())
    {
      predictiveCells.push_back(segOverlap->segment.cell);
    }
  }

  return predictiveCells;
}

vector<CellIdx> ExtendedTemporalMemory::getWinnerCells() const
{
  return winnerCells_;
}

vector<CellIdx> ExtendedTemporalMemory::getMatchingCells() const
{
  vector<CellIdx> matchingCells;

  for (auto segOverlap = matchingSegments_.begin();
       segOverlap != matchingSegments_.end();
       segOverlap++)
  {
    if (segOverlap == matchingSegments_.begin() ||
        segOverlap->segment.cell != matchingCells.back())
    {
      matchingCells.push_back(segOverlap->segment.cell);
    }
  }

  return matchingCells;
}

vector<Segment> ExtendedTemporalMemory::getActiveSegments() const
{
  vector<Segment> ret;
  ret.reserve(activeSegments_.size());
  for (const SegmentOverlap& segmentOverlap : activeSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

vector<Segment> ExtendedTemporalMemory::getMatchingSegments() const
{
  vector<Segment> ret;
  ret.reserve(matchingSegments_.size());
  for (const SegmentOverlap& segmentOverlap : matchingSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

UInt ExtendedTemporalMemory::numberOfColumns() const
{
  return numColumns_;
}

bool ExtendedTemporalMemory::_validateCell(CellIdx cell)
{
  if (cell < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell;
  return false;
}

vector<UInt> ExtendedTemporalMemory::getColumnDimensions() const
{
  return columnDimensions_;
}

UInt ExtendedTemporalMemory::getCellsPerColumn() const
{
  return cellsPerColumn_;
}

UInt ExtendedTemporalMemory::getActivationThreshold() const
{
  return activationThreshold_;
}

void ExtendedTemporalMemory::setActivationThreshold(UInt activationThreshold)
{
  activationThreshold_ = activationThreshold;
}

Permanence ExtendedTemporalMemory::getInitialPermanence() const
{
  return initialPermanence_;
}

void ExtendedTemporalMemory::setInitialPermanence(Permanence initialPermanence)
{
  initialPermanence_ = initialPermanence;
}

Permanence ExtendedTemporalMemory::getConnectedPermanence() const
{
  return connectedPermanence_;
}

void ExtendedTemporalMemory::setConnectedPermanence(
  Permanence connectedPermanence)
{
  connectedPermanence_ = connectedPermanence;
}

UInt ExtendedTemporalMemory::getMinThreshold() const
{
  return minThreshold_;
}

void ExtendedTemporalMemory::setMinThreshold(UInt minThreshold)
{
  minThreshold_ = minThreshold;
}

UInt ExtendedTemporalMemory::getMaxNewSynapseCount() const
{
  return maxNewSynapseCount_;
}

void ExtendedTemporalMemory::setMaxNewSynapseCount(UInt maxNewSynapseCount)
{
  maxNewSynapseCount_ = maxNewSynapseCount;
}

Permanence ExtendedTemporalMemory::getPermanenceIncrement() const
{
  return permanenceIncrement_;
}

void ExtendedTemporalMemory::setPermanenceIncrement(
  Permanence permanenceIncrement)
{
  permanenceIncrement_ = permanenceIncrement;
}

Permanence ExtendedTemporalMemory::getPermanenceDecrement() const
{
  return permanenceDecrement_;
}

void ExtendedTemporalMemory::setPermanenceDecrement(
  Permanence permanenceDecrement)
{
  permanenceDecrement_ = permanenceDecrement;
}

Permanence ExtendedTemporalMemory::getPredictedSegmentDecrement() const
{
  return predictedSegmentDecrement_;
}

void ExtendedTemporalMemory::setPredictedSegmentDecrement(
  Permanence predictedSegmentDecrement)
{
  predictedSegmentDecrement_ = predictedSegmentDecrement;
}

/**
* Create a RNG with given seed
*/
void ExtendedTemporalMemory::seed_(UInt64 seed)
{
  rng_ = Random(seed);
}

UInt ExtendedTemporalMemory::persistentSize() const
{
  // TODO: this won't scale!
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}

void ExtendedTemporalMemory::save(ostream& outStream) const
{
  // Write a starting marker and version.
  outStream << "ExtendedTemporalMemory" << endl;
  outStream << version_ << endl;

  outStream << numColumns_ << " "
    << cellsPerColumn_ << " "
    << activationThreshold_ << " "
    << initialPermanence_ << " "
    << connectedPermanence_ << " "
    << minThreshold_ << " "
    << maxNewSynapseCount_ << " "
    << permanenceIncrement_ << " "
    << permanenceDecrement_ << " "
    << predictedSegmentDecrement_ << " "
    << endl;

  connections.save(outStream);
  outStream << endl;

  outStream << rng_ << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto & elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << activeCells_.size() << " ";
  for (CellIdx cell : activeCells_) {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << winnerCells_.size() << " ";
  for (CellIdx cell : winnerCells_) {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << activeSegments_.size() << " ";
  for (SegmentOverlap elem : activeSegments_) {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << matchingSegments_.size() << " ";
  for (SegmentOverlap elem : matchingSegments_) {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << "~ExtendedTemporalMemory" << endl;
}

void ExtendedTemporalMemory::write(ExtendedTemporalMemoryProto::Builder& proto) const
{
  auto columnDims = proto.initColumnDimensions(columnDimensions_.size());
  for (UInt i = 0; i < columnDimensions_.size(); i++)
  {
    columnDims.set(i, columnDimensions_[i]);
  }

  proto.setCellsPerColumn(cellsPerColumn_);
  proto.setActivationThreshold(activationThreshold_);
  proto.setInitialPermanence(initialPermanence_);
  proto.setConnectedPermanence(connectedPermanence_);
  proto.setMinThreshold(minThreshold_);
  proto.setMaxNewSynapseCount(maxNewSynapseCount_);
  proto.setPermanenceIncrement(permanenceIncrement_);
  proto.setPermanenceDecrement(permanenceDecrement_);
  proto.setPredictedSegmentDecrement(predictedSegmentDecrement_);

  auto _connections = proto.initConnections();
  connections.write(_connections);

  auto random = proto.initRandom();
  rng_.write(random);

  auto activeCells = proto.initActiveCells(activeCells_.size());
  UInt i = 0;
  for (CellIdx cell : activeCells_)
  {
    activeCells.set(i++, cell);
  }

  auto activeSegmentOverlaps =
    proto.initActiveSegmentOverlaps(activeSegments_.size());
  for (UInt i = 0; i < activeSegments_.size(); ++i)
  {
    Segment segment = activeSegments_[i].segment;
    activeSegmentOverlaps[i].setCell(segment.cell);
    activeSegmentOverlaps[i].setSegment(segment.idx);
    activeSegmentOverlaps[i].setOverlap(activeSegments_[i].overlap);
  }

  auto winnerCells = proto.initWinnerCells(winnerCells_.size());
  i = 0;
  for (CellIdx cell : winnerCells_)
  {
    winnerCells.set(i++, cell);
  }

  auto matchingSegmentOverlaps =
    proto.initMatchingSegmentOverlaps(matchingSegments_.size());
  for (UInt i = 0; i < matchingSegments_.size(); ++i)
  {
    Segment segment = matchingSegments_[i].segment;
    matchingSegmentOverlaps[i].setCell(segment.cell);
    matchingSegmentOverlaps[i].setSegment(segment.idx);
    matchingSegmentOverlaps[i].setOverlap(matchingSegments_[i].overlap);
  }
}

// Implementation note: this method sets up the instance using data from
// proto. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void ExtendedTemporalMemory::read(ExtendedTemporalMemoryProto::Reader& proto)
{
  numColumns_ = 1;
  columnDimensions_.clear();
  for (UInt dimension : proto.getColumnDimensions())
  {
    numColumns_ *= dimension;
    columnDimensions_.push_back(dimension);
  }

  cellsPerColumn_ = proto.getCellsPerColumn();
  activationThreshold_ = proto.getActivationThreshold();
  initialPermanence_ = proto.getInitialPermanence();
  connectedPermanence_ = proto.getConnectedPermanence();
  minThreshold_ = proto.getMinThreshold();
  maxNewSynapseCount_ = proto.getMaxNewSynapseCount();
  permanenceIncrement_ = proto.getPermanenceIncrement();
  permanenceDecrement_ = proto.getPermanenceDecrement();
  predictedSegmentDecrement_ = proto.getPredictedSegmentDecrement();

  auto _connections = proto.getConnections();
  connections.read(_connections);

  auto random = proto.getRandom();
  rng_.read(random);

  activeCells_.clear();
  for (auto cell : proto.getActiveCells())
  {
    activeCells_.push_back(cell);
  }

  if (proto.getActiveSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "ExtendedTemporalMemory::read :: Obsolete field 'activeSegments' isn't usable. "
             << "TemporalMemory results will be goofy for one timestep.";
  }

  activeSegments_.clear();
  for (auto value : proto.getActiveSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    activeSegments_.push_back({segment, value.getOverlap()});
  }

  winnerCells_.clear();
  for (auto cell : proto.getWinnerCells())
  {
    winnerCells_.push_back(cell);
  }

  if (proto.getMatchingSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "Obsolete field 'matchingSegments' isn't usable."
             << "ExtendedTemporalMemory results will be goofy for one timestep.";
  }

  matchingSegments_.clear();
  for (auto value : proto.getMatchingSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    matchingSegments_.push_back({segment, value.getOverlap()});
  }
}

void ExtendedTemporalMemory::load(istream& inStream)
{
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "ExtendedTemporalMemory");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= version_);

  // Retrieve simple variables
  inStream >> numColumns_
    >> cellsPerColumn_
    >> activationThreshold_
    >> initialPermanence_
    >> connectedPermanence_
    >> minThreshold_
    >> maxNewSynapseCount_
    >> permanenceIncrement_
    >> permanenceDecrement_
    >> predictedSegmentDecrement_;

  connections.load(inStream);

  inStream >> rng_;

  // Retrieve vectors.
  UInt numColumnDimensions;
  inStream >> numColumnDimensions;
  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++)
  {
    inStream >> columnDimensions_[i];
  }

  UInt numActiveCells;
  inStream >> numActiveCells;
  for (UInt i = 0; i < numActiveCells; i++)
  {
    CellIdx cell;
    inStream >> cell;
    activeCells_.push_back(cell);
  }

  if (version < 2)
  {
    UInt numPredictiveCells;
    inStream >> numPredictiveCells;
    for (UInt i = 0; i < numPredictiveCells; i++)
    {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  if (version < 2)
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments_.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments_[i].segment.idx;
      inStream >> activeSegments_[i].segment.cell;
      activeSegments_[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments_.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments_[i].segment.idx;
      inStream >> activeSegments_[i].segment.cell;
      inStream >> activeSegments_[i].overlap;
    }
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++)
  {
    CellIdx cell;
    inStream >> cell;
    winnerCells_.push_back(cell);
  }

  if (version < 2)
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments_.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments_[i].segment.idx;
      inStream >> matchingSegments_[i].segment.cell;
      matchingSegments_[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments_.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments_[i].segment.idx;
      inStream >> matchingSegments_[i].segment.cell;
      inStream >> matchingSegments_[i].overlap;
    }
  }

  if (version < 2)
  {
    UInt numMatchingCells;
    inStream >> numMatchingCells;
    for (UInt i = 0; i < numMatchingCells; i++) {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  inStream >> marker;
  NTA_CHECK(marker == "~ExtendedTemporalMemory");

}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main TM creation parameters
void ExtendedTemporalMemory::printParameters()
{
  std::cout << "------------CPP ExtendedTemporalMemory Parameters ------------------\n";
  std::cout
    << "version                   = " << version_ << std::endl
    << "numColumns                = " << numberOfColumns() << std::endl
    << "cellsPerColumn            = " << getCellsPerColumn() << std::endl
    << "activationThreshold       = " << getActivationThreshold() << std::endl
    << "initialPermanence         = " << getInitialPermanence() << std::endl
    << "connectedPermanence       = " << getConnectedPermanence() << std::endl
    << "minThreshold              = " << getMinThreshold() << std::endl
    << "maxNewSynapseCount        = " << getMaxNewSynapseCount() << std::endl
    << "permanenceIncrement       = " << getPermanenceIncrement() << std::endl
    << "permanenceDecrement       = " << getPermanenceDecrement() << std::endl
    << "predictedSegmentDecrement = " << getPredictedSegmentDecrement() << std::endl;
}

void ExtendedTemporalMemory::printState(vector<UInt> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void ExtendedTemporalMemory::printState(vector<Real> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::printf("%6.3f ", state[i]);
  }
  std::cout << "]\n";
}
