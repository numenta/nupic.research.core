# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import _nupic


try:
    # NOTE need to import capnp first to activate the magic necessary for
    # RandomProto_capnp
    import capnp
except ImportError:
    capnp = None
else:
    from nupic.proto.RandomProto_capnp import RandomProto


# Capnp reader traveral limit (see capnp::ReaderOptions)
_TRAVERSAL_LIMIT_IN_WORDS = 1 << 63


class Random(_nupic.Random):
    def write(self, pyBuilder):
      """Serialize the Random instance using capnp.

      :param: Destination RandomProto message builder
      """
      reader = RandomProto.from_bytes(
          self._writeAsCapnpPyBytes(),
          traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
      pyBuilder.from_dict(reader.to_dict())  # copy


    def read(self, proto):
      """Initialize the Random instance from the given RandomProto reader.

      :param proto: RandomProto message reader containing data from a previously
                    serialized Random instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
