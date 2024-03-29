/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * Implementation of unit tests for ApicalTiebreakTemporalMemory
 */

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

#include <nupic/algorithms/ApicalTiebreakTemporalMemory.hpp>
#include "gtest/gtest.h"

using namespace nupic;
using namespace nupic::algorithms::apical_tiebreak_temporal_memory;
using namespace std;

#define EPSILON 0.0000001

namespace {

  TEST(ApicalTiebreakTemporalMemoryTest, testInitInvalidParams)
  {
    // Invalid columnCount
    EXPECT_THROW(ApicalTiebreakSequenceMemory(
                   /*columnCount*/ 0,
                   /*apicalInputSize*/ 0,
                   /*cellsPerColumn*/ 32),
                 exception);

    // Invalid cellsPerColumn
    EXPECT_THROW(ApicalTiebreakSequenceMemory(
                   /*columnCount*/ 32,
                   /*apicalInputSize*/ 0,
                   /*cellsPerColumn*/ 0),
                 exception);
  }

  /**
   * When a predicted column is activated, only the predicted cells in the
   * columns should be activated.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ActivateCorrectlyPredictiveCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment activeSegment =
      tm.createBasalSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);
    ASSERT_EQ(expectedActiveCells, tm.getPredictedCells());

    EXPECT_EQ(expectedActiveCells, tm.getActiveCells());
  }

  /**
   * When an unpredicted column is activated, every cell in the column should
   * become active.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, BurstUnpredictedColumns)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> activeColumns = {0};
    const vector<CellIdx> burstingCells = {0, 1, 2, 3};

    tm.compute(activeColumns);

    EXPECT_EQ(burstingCells, tm.getActiveCells());
  }

  /**
   * When the ApicalTiebreakTemporalMemory receives zero active columns, it should still
   * compute the active cells, winner cells, and predictive cells. All should be
   * empty.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ZeroActiveColumns)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Make some cells predictive.
    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};

    Segment segment = tm.createBasalSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(segment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(segment, previousActiveCells[3], 0.5);

    tm.compute(previousActiveColumns);
    ASSERT_FALSE(tm.getActiveCells().empty());
    ASSERT_FALSE(tm.getWinnerCells().empty());

    const vector<UInt> zeroColumns = {};
    tm.compute(zeroColumns);

    EXPECT_TRUE(tm.getActiveCells().empty());
    EXPECT_TRUE(tm.getWinnerCells().empty());
  }

  /**
   * All predicted active cells are winner cells, even when learning is
   * disabled.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, PredictedActiveCellsAreAlwaysWinners)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedWinnerCells = {4, 6};

    Segment activeSegment1 =
      tm.createBasalSegment(expectedWinnerCells[0]);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment1, previousActiveCells[2], 0.5);

    Segment activeSegment2 =
      tm.createBasalSegment(expectedWinnerCells[1]);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment2, previousActiveCells[2], 0.5);

    tm.compute(previousActiveColumns, {}, {}, false);
    tm.compute(activeColumns, {}, {}, false);

    EXPECT_EQ(expectedWinnerCells, tm.getWinnerCells());
  }

  /**
   * One cell in each bursting column is a winner cell, even when learning is
   * disabled.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ChooseOneWinnerCellInBurstingColumn)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> activeColumns = {0};
    const set<CellIdx> burstingCells = {0, 1, 2, 3};

    tm.compute(activeColumns, {}, {}, false);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    EXPECT_TRUE(burstingCells.find(winnerCells[0]) != burstingCells.end());
  }

  /**
   * Active segments on predicted active cells should be reinforced. Active
   * synapses should be reinforced, inactive synapses should be punished.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ReinforceCorrectlyActiveSegments)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.08,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> activeCells = {5};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.createBasalSegment(activeCell);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(activeSegment, 81, 0.5);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.6, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.42, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * The best matching segment in a bursting column should be reinforced. Active
   * synapses should be strengthened, and inactive synapses should be weakened.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ReinforceSelectedMatchingSegmentInBurstingColumn)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.08,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.createBasalSegment(burstingCells[0]);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[1], 0.3);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        previousActiveCells[2], 0.3);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(selectedMatchingSegment,
                                        81, 0.3);

    // Add some competition.
    Segment otherMatchingSegment =
      tm.createBasalSegment(burstingCells[1]);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      previousActiveCells[0], 0.3);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      previousActiveCells[1], 0.3);
    tm.basalConnections.createSynapse(otherMatchingSegment,
                                      81, 0.3);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.4, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.22, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a column bursts, don't reward or punish matching-but-not-selected
   * segments.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, NoChangeToNonselectedMatchingSegmentsInBurstingColumn)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.08,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> burstingCells = {4, 5, 6, 7};

    Segment selectedMatchingSegment =
      tm.createBasalSegment(burstingCells[0]);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[0], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[1], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      previousActiveCells[2], 0.3);
    tm.basalConnections.createSynapse(selectedMatchingSegment,
                                      81, 0.3);

    Segment otherMatchingSegment =
      tm.createBasalSegment(burstingCells[1]);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        previousActiveCells[0], 0.3);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        previousActiveCells[1], 0.3);
    Synapse inactiveSynapse =
      tm.basalConnections.createSynapse(otherMatchingSegment,
                                        81, 0.3);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(inactiveSynapse).permanence,
                EPSILON);
  }

  /**
   * When a predicted column is activated, don't reward or punish
   * matching-but-not-active segments anywhere in the column.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, NoChangeToMatchingSegmentsInPredictedActiveColumn)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<UInt> activeColumns = {1};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<CellIdx> expectedActiveCells = {4};
    const vector<CellIdx> otherBurstingCells = {5, 6, 7};

    Segment activeSegment =
      tm.createBasalSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.5);

    Segment matchingSegmentOnSameCell =
      tm.createBasalSegment(expectedActiveCells[0]);
    Synapse synapse1 =
      tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                        previousActiveCells[0], 0.3);
    Synapse synapse2 =
      tm.basalConnections.createSynapse(matchingSegmentOnSameCell,
                                        previousActiveCells[1], 0.3);

    Segment matchingSegmentOnOtherCell =
      tm.createBasalSegment(otherBurstingCells[0]);
    Synapse synapse3 =
      tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                        previousActiveCells[0], 0.3);
    Synapse synapse4 =
      tm.basalConnections.createSynapse(matchingSegmentOnOtherCell,
                                        previousActiveCells[1], 0.3);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);
    ASSERT_EQ(expectedActiveCells, tm.getPredictedCells());

    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.3, tm.basalConnections.dataForSynapse(synapse4).permanence,
                EPSILON);
  }

  /**
   * When growing a new segment, if there are no previous winner cells, don't
   * even grow the segment. It will never match.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, NoNewSegmentIfNotEnoughWinnerCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 2,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> zeroColumns = {};
    const vector<UInt> activeColumns = {0};

    tm.compute(zeroColumns);
    tm.compute(activeColumns);

    EXPECT_EQ(0, tm.basalConnections.numSegments());
  }

  /**
   * When growing a new segment, if the number of previous winner cells is above
   * sampleSize, grow sampleSize synapses.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, NewSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 2,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns= {0, 1, 2};
    const vector<UInt> activeColumns = {4};

    tm.compute(previousActiveColumns);

    vector<CellIdx> prevWinnerCells = tm.getWinnerCells();
    ASSERT_EQ(3, prevWinnerCells.size());

    tm.compute(activeColumns);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    vector<Segment> segments = tm.basalConnections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
    ASSERT_EQ(2, synapses.size());
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[0] ||
                  synapseData.presynapticCell == prevWinnerCells[1] ||
                  synapseData.presynapticCell == prevWinnerCells[2]);
    }

  }

  /**
   * When growing a new segment, if the number of previous winner cells is below
   * sampleSize, grow synapses to all of the previous winner cells.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, NewSegmentAddSynapsesToAllWinnerCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0, 1, 2};
    const vector<UInt> activeColumns = {4};

    tm.compute(previousActiveColumns);

    vector<CellIdx> prevWinnerCells = tm.getWinnerCells();
    ASSERT_EQ(3, prevWinnerCells.size());

    tm.compute(activeColumns);

    vector<CellIdx> winnerCells = tm.getWinnerCells();
    ASSERT_EQ(1, winnerCells.size());
    vector<Segment> segments = tm.basalConnections.segmentsForCell(winnerCells[0]);
    ASSERT_EQ(1, segments.size());
    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
    ASSERT_EQ(3, synapses.size());

    vector<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      presynapticCells.push_back(synapseData.presynapticCell);
    }
    std::sort(presynapticCells.begin(), presynapticCells.end());
    EXPECT_EQ(prevWinnerCells, presynapticCells);
  }

  /**
   * When adding synapses to a matching segment, the final number of active
   * synapses on the segment should be sampleSize, assuming there are
   * enough previous winner cells available to connect to.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, MatchingSegmentAddSynapsesToSubsetOfWinnerCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const vector<UInt> previousActiveColumns = {0, 1, 2, 3};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {4};

    Segment matchingSegment = tm.createBasalSegment(4);
    tm.basalConnections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(3, synapses.size());
    for (SynapseIdx i = 1; i < synapses.size(); i++)
    {
      SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[i]);
      EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
      EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[1] ||
                  synapseData.presynapticCell == prevWinnerCells[2] ||
                  synapseData.presynapticCell == prevWinnerCells[3]);
    }
  }

  /**
   * When adding synapses to a matching segment, if the number of previous
   * winner cells is lower than (sampleSize - nActiveSynapsesOnSegment),
   * grow synapses to all the previous winner cells.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, MatchingSegmentAddSynapsesToAllWinnerCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const vector<UInt> previousActiveColumns = {0, 1};
    const vector<CellIdx> prevWinnerCells = {0, 1};
    const vector<UInt> activeColumns = {4};

    Segment matchingSegment = tm.createBasalSegment(4);
    tm.basalConnections.createSynapse(matchingSegment, 0, 0.5);

    tm.compute(previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(2, synapses.size());

    SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[1]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_EQ(prevWinnerCells[1], synapseData.presynapticCell);
  }

  /**
   * When a segment becomes active, grow synapses to previous winner cells.
   *
   * The number of grown synapses is calculated from the "matching segment"
   * overlap, not the "active segment" overlap.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ActiveSegmentGrowSynapsesAccordingToPotentialOverlap)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 2,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const vector<UInt> previousActiveColumns = {0, 1, 2, 3, 4};
    const vector<CellIdx> prevWinnerCells = {0, 1, 2, 3, 4};
    const vector<UInt> activeColumns = {5};

    Segment activeSegment = tm.createBasalSegment(5);
    tm.basalConnections.createSynapse(activeSegment, 0, 0.5);
    tm.basalConnections.createSynapse(activeSegment, 1, 0.5);
    tm.basalConnections.createSynapse(activeSegment, 2, 0.2);

    tm.compute(previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    tm.compute(activeColumns);

    vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(activeSegment);

    ASSERT_EQ(4, synapses.size());

    SynapseData synapseData = tm.basalConnections.dataForSynapse(synapses[3]);
    EXPECT_NEAR(0.21, synapseData.permanence, EPSILON);
    EXPECT_TRUE(synapseData.presynapticCell == prevWinnerCells[3] ||
                synapseData.presynapticCell == prevWinnerCells[4]);
  }

  /**
   * When a synapse is punished for contributing to a wrong prediction, if its
   * permanence falls to 0 it should be destroyed.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, DestroyWeakSynapseOnWrongPrediction)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {2};
    const CellIdx expectedActiveCell = 5;

    Segment activeSegment = tm.createBasalSegment(expectedActiveCell);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);

    // Weak synapse.
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[3], 0.015);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_EQ(3, tm.basalConnections.numSynapses(activeSegment));
  }

  /**
   * When a synapse is punished for not contributing to a right prediction, if
   * its permanence falls to 0 it should be destroyed.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, DestroyWeakSynapseOnActiveReinforce)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {1};
    const CellIdx activeCell = 5;

    Segment activeSegment = tm.createBasalSegment(activeCell);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);

    // Weak inactive synapse.
    tm.basalConnections.createSynapse(activeSegment, 81, 0.09);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_EQ(3, tm.basalConnections.numSynapses(activeSegment));
  }

  /**
   * When a segment adds synapses and it runs over maxSynapsesPerSegment, it
   * should make room by destroying synapses with the lowest permanence.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, RecycleWeakestSynapseToMakeRoomForNewSynapse)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.21,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 1,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 255,
      /*maxSynapsesPerSegment*/ 4
      );

    // Use 1 cell per column so that we have easy control over the winner cells.
    const vector<UInt> previousActiveColumns= {1, 2, 3};
    const vector<CellIdx> prevWinnerCells = {1, 2, 3};
    const vector<UInt> activeColumns = {4};

    Segment matchingSegment = tm.createBasalSegment(4);

    // Create a weak synapse. Make sure it's not so weak that
    // permanenceDecrement destroys it.
    tm.basalConnections.createSynapse(matchingSegment, 0, 0.11);

    // Create a synapse that will match.
    tm.basalConnections.createSynapse(matchingSegment, 1, 0.20);

    // Create a synapse with a high permanence.
    tm.basalConnections.createSynapse(matchingSegment, 31, 0.6);

    // Activate a synapse on the segment, making it "matching".
    tm.compute(previousActiveColumns);

    ASSERT_EQ(prevWinnerCells, tm.getWinnerCells());

    // Now mark the segment as "correct" by activating its cell.
    tm.compute(activeColumns);

    // There should now be 3 synapses, and none of them should be to cell 0.
    const vector<Synapse>& synapses =
      tm.basalConnections.synapsesForSegment(matchingSegment);
    ASSERT_EQ(4, synapses.size());

    std::set<CellIdx> presynapticCells;
    for (Synapse synapse : synapses)
    {
      presynapticCells.insert(
        tm.basalConnections.dataForSynapse(synapse).presynapticCell);
    }

    std::set<CellIdx> expected = {1, 2, 3, 31};
    EXPECT_EQ(expected, presynapticCells);
  }

  /**
   * When a cell adds a segment and it runs over maxSegmentsPerCell, it should
   * make room by destroying the least recently active segment.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, RecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 1,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.50,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 3,
      /*permanenceIncrement*/ 0.02,
      /*permanenceDecrement*/ 0.02,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42,
      /*maxSegmentsPerCell*/ 2
      );

    const vector<UInt> previousActiveColumns1 = {0, 1, 2};
    const vector<UInt> previousActiveColumns2 = {3, 4, 5};
    const vector<UInt> previousActiveColumns3 = {6, 7, 8};
    const vector<UInt> activeColumns = {9};

    tm.compute(previousActiveColumns1);
    tm.compute(activeColumns);

    ASSERT_EQ(1, tm.basalConnections.numSegments(9));
    Segment oldestSegment = tm.basalConnections.segmentsForCell(9)[0];

    tm.reset();
    tm.compute(previousActiveColumns2);
    tm.compute(activeColumns);

    ASSERT_EQ(2, tm.basalConnections.numSegments(9));

    set<CellIdx> oldPresynaptic;
    for (Synapse synapse : tm.basalConnections.synapsesForSegment(oldestSegment))
    {
      oldPresynaptic.insert(
        tm.basalConnections.dataForSynapse(synapse).presynapticCell);
    }

    tm.reset();
    tm.compute(previousActiveColumns3);
    tm.compute(activeColumns);

    ASSERT_EQ(2, tm.basalConnections.numSegments(9));

    // Verify none of the segments are connected to the cells the old segment
    // was connected to.

    for (Segment segment : tm.basalConnections.segmentsForCell(9))
    {
      set<CellIdx> newPresynaptic;
      for (Synapse synapse : tm.basalConnections.synapsesForSegment(segment))
      {
        newPresynaptic.insert(
          tm.basalConnections.dataForSynapse(synapse).presynapticCell);
      }

      vector<CellIdx> intersection;
      std::set_intersection(oldPresynaptic.begin(), oldPresynaptic.end(),
                            newPresynaptic.begin(), newPresynaptic.end(),
                            std::back_inserter(intersection));

      vector<CellIdx> expected = {};
      EXPECT_EQ(expected, intersection);
    }
  }

  /**
   * When a segment's number of synapses falls to 0, the segment should be
   * destroyed.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, DestroySegmentsWithTooFewSynapsesToBeMatching)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {2};
    const CellIdx expectedActiveCell = 5;

    Segment matchingSegment = tm.createBasalSegment(expectedActiveCell);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[0], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[1], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[2], 0.015);
    tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[3], 0.015);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_EQ(0, tm.basalConnections.numSegments(expectedActiveCell));
  }

  /**
   * When a column with a matching segment isn't activated, punish the matching
   * segment.
   *
   * To exercise the implementation:
   *
   *  - Use cells before, between, and after the active columns.
   *  - Use segments that are matching-but-not-active and matching-and-active.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, PunishMatchingSegmentsInInactiveColumns)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {1};
    const CellIdx previousInactiveCell = 81;

    Segment activeSegment = tm.createBasalSegment(42);
    Synapse activeSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse2 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[1], 0.5);
    Synapse activeSynapse3 =
      tm.basalConnections.createSynapse(activeSegment, previousActiveCells[2], 0.5);
    Synapse inactiveSynapse1 =
      tm.basalConnections.createSynapse(activeSegment, previousInactiveCell, 0.5);

    Segment matchingSegment = tm.createBasalSegment(43);
    Synapse activeSynapse4 =
      tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[0], 0.5);
    Synapse activeSynapse5 =
      tm.basalConnections.createSynapse(matchingSegment, previousActiveCells[1], 0.5);
    Synapse inactiveSynapse2 =
      tm.basalConnections.createSynapse(matchingSegment, previousInactiveCell, 0.5);

    tm.compute(previousActiveColumns);
    tm.compute(activeColumns);

    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse2).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse3).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse4).permanence,
                EPSILON);
    EXPECT_NEAR(0.48, tm.basalConnections.dataForSynapse(activeSynapse5).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.basalConnections.dataForSynapse(inactiveSynapse1).permanence,
                EPSILON);
    EXPECT_NEAR(0.50, tm.basalConnections.dataForSynapse(inactiveSynapse2).permanence,
                EPSILON);
  }

  /**
   * In a bursting column with no matching segments, a segment should be added
   * to the cell with the fewest segments. When there's a tie, choose randomly.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, AddSegmentToCellWithFewestSegments)
  {
    bool grewOnCell1 = false;
    bool grewOnCell2 = false;
    for (UInt seed = 0; seed < 100; seed++)
    {
      ApicalTiebreakSequenceMemory tm(
        /*columnCount*/ 32,
        /*apicalInputSize*/ 0,
        /*cellsPerColumn*/ 4,
        /*activationThreshold*/ 3,
        /*initialPermanence*/ 0.2,
        /*connectedPermanence*/ 0.50,
        /*minThreshold*/ 2,
        /*sampleSize*/ 4,
        /*permanenceIncrement*/ 0.10,
        /*permanenceDecrement*/ 0.10,
        /*basalPredictedSegmentDecrement*/ 0.02,
        /*apicalPredictedSegmentDecrement*/ 0.0,
        /*learnOnOneCell*/ false,
        /*seed*/ seed
        );

      // enough for 4 winner cells
      const vector<UInt> previousActiveColumns = {1, 2, 3, 4};
      const vector<UInt> activeColumns = {0};
      const vector<CellIdx> previousActiveCells =
        {4, 5, 6, 7}; // (there are more)
      vector<CellIdx> nonmatchingCells = {0, 3};
      vector<CellIdx> activeCells = {0, 1, 2, 3};

      Segment segment1 = tm.createBasalSegment(nonmatchingCells[0]);
      tm.basalConnections.createSynapse(segment1, previousActiveCells[0], 0.5);
      Segment segment2 = tm.createBasalSegment(nonmatchingCells[1]);
      tm.basalConnections.createSynapse(segment2, previousActiveCells[1], 0.5);

      tm.compute(previousActiveColumns);
      tm.compute(activeColumns);

      ASSERT_EQ(activeCells, tm.getActiveCells());

      EXPECT_EQ(3, tm.basalConnections.numSegments());
      EXPECT_EQ(1, tm.basalConnections.segmentsForCell(0).size());
      EXPECT_EQ(1, tm.basalConnections.segmentsForCell(3).size());
      EXPECT_EQ(1, tm.basalConnections.numSynapses(segment1));
      EXPECT_EQ(1, tm.basalConnections.numSynapses(segment2));

      vector<Segment> segments = tm.basalConnections.segmentsForCell(1);
      if (segments.empty())
      {
        vector<Segment> segments2 = tm.basalConnections.segmentsForCell(2);
        EXPECT_FALSE(segments2.empty());
        grewOnCell2 = true;
        segments.insert(segments.end(), segments2.begin(), segments2.end());
      }
      else
      {
        grewOnCell1 = true;
      }

      ASSERT_EQ(1, segments.size());
      vector<Synapse> synapses = tm.basalConnections.synapsesForSegment(segments[0]);
      EXPECT_EQ(4, synapses.size());

      set<CellIdx> columnChecklist(previousActiveColumns.begin(),
                                   previousActiveColumns.end());

      for (Synapse synapse : synapses)
      {
        SynapseData synapseData = tm.basalConnections.dataForSynapse(synapse);
        EXPECT_NEAR(0.2, synapseData.permanence, EPSILON);

        UInt32 column = (UInt)tm.columnForCell(synapseData.presynapticCell);
        auto position = columnChecklist.find(column);
        EXPECT_NE(columnChecklist.end(), position);
        columnChecklist.erase(position);
      }
      EXPECT_TRUE(columnChecklist.empty());
    }

    EXPECT_TRUE(grewOnCell1);
    EXPECT_TRUE(grewOnCell2);
  }

  /**
   * When the best matching segment has more than sampleSize matching
   * synapses, don't grow new synapses. This test is specifically aimed at
   * unexpected behavior with negative numbers and unsigned integers.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, SampleSizeOverflow)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    Segment segment = tm.createBasalSegment(8);
    tm.basalConnections.createSynapse(segment, 0, 0.2);
    tm.basalConnections.createSynapse(segment, 1, 0.2);
    tm.basalConnections.createSynapse(segment, 2, 0.2);
    tm.basalConnections.createSynapse(segment, 3, 0.2);
    tm.basalConnections.createSynapse(segment, 4, 0.2);
    Synapse sampleSynapse = tm.basalConnections.createSynapse(segment, 5, 0.2);
    tm.basalConnections.createSynapse(segment, 6, 0.2);
    tm.basalConnections.createSynapse(segment, 7, 0.2);

    const vector<UInt> previousActiveColumns = {0, 1, 3, 4};
    tm.compute(previousActiveColumns);

    ASSERT_EQ(1, tm.getMatchingBasalSegments().size());

    const vector<UInt> activeColumns = {2};
    tm.compute(activeColumns);

    // Make sure the segment has learned.
    ASSERT_NEAR(0.3, tm.basalConnections.dataForSynapse(sampleSynapse).permanence,
                EPSILON);

    EXPECT_EQ(8, tm.basalConnections.numSynapses(segment));
  }

  /**
   * With learning disabled, generate some predicted active columns, predicted
   * inactive columns, and nonpredicted active columns. The connections should
   * not change.
   */
  TEST(ApicalTiebreakTemporalMemoryTest, ConnectionsNeverChangeWhenLearningDisabled)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 32,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 3,
      /*initialPermanence*/ 0.2,
      /*connectedPermanence*/ 0.50,
      /*minThreshold*/ 2,
      /*sampleSize*/ 4,
      /*permanenceIncrement*/ 0.10,
      /*permanenceDecrement*/ 0.10,
      /*basalPredictedSegmentDecrement*/ 0.02,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42
      );

    const vector<UInt> previousActiveColumns = {0};
    const vector<CellIdx> previousActiveCells = {0, 1, 2, 3};
    const vector<UInt> activeColumns = {
      1, // predicted
      2  // bursting
    };
    const CellIdx previousInactiveCell = 81;
    const vector<CellIdx> expectedActiveCells = {4};

    Segment correctActiveSegment =
      tm.createBasalSegment(expectedActiveCells[0]);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(correctActiveSegment,
                                      previousActiveCells[2], 0.5);

    Segment wrongMatchingSegment = tm.createBasalSegment(43);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousActiveCells[0], 0.5);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousActiveCells[1], 0.5);
    tm.basalConnections.createSynapse(wrongMatchingSegment,
                                      previousInactiveCell, 0.5);

    Connections before = tm.basalConnections;

    tm.compute(previousActiveColumns,
                             {}, {},
                             false);
    tm.compute(activeColumns,
                             {}, {},
                             false);

    EXPECT_EQ(before, tm.basalConnections);
  }

  TEST(ApicalTiebreakTemporalMemoryTest, testColumnForCell)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 2048,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 5);

    ASSERT_EQ(0, tm.columnForCell(0));
    ASSERT_EQ(0, tm.columnForCell(4));
    ASSERT_EQ(1, tm.columnForCell(5));
    ASSERT_EQ(2047, tm.columnForCell(10239));
  }

  TEST(ApicalTiebreakTemporalMemoryTest, testColumnForCellInvalidCell)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 4096,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4);

    EXPECT_NO_THROW(tm.columnForCell(16383));
    EXPECT_THROW(tm.columnForCell(16384), std::exception);
    EXPECT_THROW(tm.columnForCell(-1), std::exception);
  }

  TEST(ApicalTiebreakTemporalMemoryTest, testNumberOfColumns)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 2048,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 32);

    int numOfColumns = tm.numberOfColumns();
    ASSERT_EQ(numOfColumns, 2048);
  }

  TEST(ApicalTiebreakTemporalMemoryTest, testNumberOfCells)
  {
    ApicalTiebreakSequenceMemory tm(
      /*columnCount*/ 2048,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 32);

    Int numberOfCells = tm.numberOfCells();
    ASSERT_EQ(numberOfCells, 2048 * 32);
  }

  TEST(ApicalTiebreakTemporalMemoryTest, testWrite)
  {
    ApicalTiebreakSequenceMemory tm1(
      /*columnCount*/ 100,
      /*apicalInputSize*/ 0,
      /*cellsPerColumn*/ 4,
      /*activationThreshold*/ 7,
      /*initialPermanence*/ 0.37,
      /*connectedPermanence*/ 0.58,
      /*minThreshold*/ 4,
      /*sampleSize*/ 18,
      /*permanenceIncrement*/ 0.23,
      /*permanenceDecrement*/ 0.08,
      /*basalPredictedSegmentDecrement*/ 0.0,
      /*apicalPredictedSegmentDecrement*/ 0.0,
      /*learnOnOneCell*/ false,
      /*seed*/ 42);

    // Run some data through before serializing
    /*
      PatternMachine patternMachine = PatternMachine(100, 4);
      SequenceMachine sequenceMachine = SequenceMachine(self.patternMachine);
      Sequence sequence = self.sequenceMachine.generateFromNumbers(range(5));
    */
    vector<vector<UInt>> sequence =
      {
        { 45, 53, 70, 83 },
        { 8, 59, 65, 67 },
        { 25, 39, 98, 99 },
        { 11, 14, 66, 78 },
        { 69, 87, 95, 96 } };

    for (UInt i = 0; i < 3; i++)
    {
      for (vector<UInt> pattern : sequence)
        tm1.compute(pattern);
    }

    // Write and read back the proto
    stringstream ss;
    tm1.write(ss);
    ApicalTiebreakSequenceMemory tm2;
    tm2.read(ss);

    // Check that the two temporal memory objects have the same attributes
    ASSERT_TRUE(tm1 == tm2);

    tm1.compute(sequence[0]);
    tm2.compute(sequence[0]);
    ASSERT_EQ(tm1.getActiveCells(), tm2.getActiveCells());
    ASSERT_EQ(tm1.getWinnerCells(), tm2.getWinnerCells());
    ASSERT_EQ(tm1.basalConnections, tm2.basalConnections);

    tm1.compute(sequence[3]);
    tm2.compute(sequence[3]);
    ASSERT_EQ(tm1.getActiveCells(), tm2.getActiveCells());

    const vector<Segment> activeSegments1 = tm1.getActiveBasalSegments();
    const vector<Segment> activeSegments2 = tm2.getActiveBasalSegments();
    ASSERT_EQ(activeSegments1.size(), activeSegments2.size());
    for (size_t i = 0; i < activeSegments1.size(); i++)
    {
      const SegmentData& segmentData1 =
        tm1.basalConnections.dataForSegment(activeSegments1[i]);
      const SegmentData& segmentData2 =
        tm2.basalConnections.dataForSegment(activeSegments2[i]);

      ASSERT_EQ(segmentData1.cell, segmentData2.cell);
    }

    const vector<Segment> matchingSegments1 = tm1.getMatchingBasalSegments();
    const vector<Segment> matchingSegments2 = tm2.getMatchingBasalSegments();
    ASSERT_EQ(matchingSegments1.size(), matchingSegments2.size());
    for (size_t i = 0; i < matchingSegments1.size(); i++)
    {
      const SegmentData& segmentData1 =
        tm1.basalConnections.dataForSegment(matchingSegments1[i]);
      const SegmentData& segmentData2 =
        tm2.basalConnections.dataForSegment(matchingSegments2[i]);

      ASSERT_EQ(segmentData1.cell, segmentData2.cell);
    }

    ASSERT_EQ(tm1.getWinnerCells(), tm2.getWinnerCells());
    ASSERT_EQ(tm1.basalConnections, tm2.basalConnections);

    ASSERT_TRUE(tm1 == tm2);
  }
}
